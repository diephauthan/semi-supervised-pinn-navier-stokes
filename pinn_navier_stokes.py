"""
Author: Than Diep
Co-authors: Mehdi Badri, Minh Doan
Affiliation: Faculty of Mechanical Engineering, HCMUT, VNU-HCM
Email: diephauthan@gmail.com
Date: 2025
License: MIT
Funding: VNU-HCM Grant C2025-20-34
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
from datetime import datetime
from typing import Tuple, List, Optional, Dict
import warnings


warnings.filterwarnings('ignore')

np.random.seed(1234)
torch.manual_seed(1234)


class PINN(nn.Module): 
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        self.layers_list = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers_list.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            nn.init.xavier_uniform_(self.layers_list[-1].weight)
            nn.init.zeros_(self.layers_list[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers_list[:-1]):
            x = torch.tanh(layer(x))
        x = self.layers_list[-1](x)
        return x


class PINN_ICBC:
    def __init__(self, 
                 x_ic: np.ndarray, y_ic: np.ndarray, t_ic: np.ndarray, 
                 u_ic: np.ndarray, v_ic: np.ndarray,
                 x_bc: np.ndarray, y_bc: np.ndarray, t_bc: np.ndarray, 
                 u_bc: np.ndarray, v_bc: np.ndarray,
                 x_unlabeled: np.ndarray, y_unlabeled: np.ndarray, 
                 t_unlabeled: np.ndarray,
                 layers: List[int], 
                 alpha_ic: float = 1.0, 
                 alpha_bc: float = 1.0, 
                 alpha_physics: float = 1.0, 
                 device: str = 'cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Loss weights
        self.alpha_ic = alpha_ic
        self.alpha_bc = alpha_bc
        self.alpha_physics = alpha_physics
        
        # Compute normalization bounds
        X_all = np.concatenate([
            np.concatenate([x_ic, y_ic, t_ic], 1),
            np.concatenate([x_bc, y_bc, t_bc], 1),
            np.concatenate([x_unlabeled, y_unlabeled, t_unlabeled], 1)
        ], 0)
        self.lb = torch.tensor(X_all.min(0), dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(X_all.max(0), dtype=torch.float32).to(self.device)
        
        # Convert data to tensors and move to device
        self._prepare_data(x_ic, y_ic, t_ic, u_ic, v_ic,
                          x_bc, y_bc, t_bc, u_bc, v_bc,
                          x_unlabeled, y_unlabeled, t_unlabeled)
        
        # Initialize network
        self.model = PINN(layers).to(self.device)
        
        # Learnable PDE parameters (Reynolds number related)
        self.lambda_1 = nn.Parameter(torch.zeros(1, device=self.device))
        self.lambda_2 = nn.Parameter(torch.zeros(1, device=self.device))
        
        # Optimizers
        self._setup_optimizers()
        
        # Loss history
        self.loss_history = {
            'iteration': [],
            'total': [],
            'ic': [],
            'bc': [],
            'physics': [],
            'lambda_1': [],
            'lambda_2': []
        }
    
    def _prepare_data(self, x_ic, y_ic, t_ic, u_ic, v_ic,
                     x_bc, y_bc, t_bc, u_bc, v_bc,
                     x_unlabeled, y_unlabeled, t_unlabeled):
        # Initial condition data
        self.x_ic = torch.tensor(x_ic, dtype=torch.float32, requires_grad=True).to(self.device)
        self.y_ic = torch.tensor(y_ic, dtype=torch.float32, requires_grad=True).to(self.device)
        self.t_ic = torch.tensor(t_ic, dtype=torch.float32, requires_grad=True).to(self.device)
        self.u_ic = torch.tensor(u_ic, dtype=torch.float32).to(self.device)
        self.v_ic = torch.tensor(v_ic, dtype=torch.float32).to(self.device)
        
        # Boundary condition data
        self.x_bc = torch.tensor(x_bc, dtype=torch.float32, requires_grad=True).to(self.device)
        self.y_bc = torch.tensor(y_bc, dtype=torch.float32, requires_grad=True).to(self.device)
        self.t_bc = torch.tensor(t_bc, dtype=torch.float32, requires_grad=True).to(self.device)
        self.u_bc = torch.tensor(u_bc, dtype=torch.float32).to(self.device)
        self.v_bc = torch.tensor(v_bc, dtype=torch.float32).to(self.device)
        
        # Unlabeled collocation points
        self.x_unlabeled = torch.tensor(x_unlabeled, dtype=torch.float32, requires_grad=True).to(self.device)
        self.y_unlabeled = torch.tensor(y_unlabeled, dtype=torch.float32, requires_grad=True).to(self.device)
        self.t_unlabeled = torch.tensor(t_unlabeled, dtype=torch.float32, requires_grad=True).to(self.device)
        
        print(f"\nData statistics:")
        print(f"  Initial Condition points (t=0): {len(x_ic)}")
        print(f"  Boundary Condition points: {len(x_bc)}")
        print(f"  Unlabeled physics points: {len(x_unlabeled)}")
        print(f"  Total supervised points: {len(x_ic) + len(x_bc)}")
        print(f"  Total points: {len(x_ic) + len(x_bc) + len(x_unlabeled)}")
    
    def _setup_optimizers(self):
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [self.lambda_1, self.lambda_2],
            lr=1e-3
        )
        
        self.optimizer_lbfgs = torch.optim.LBFGS(
            list(self.model.parameters()) + [self.lambda_1, self.lambda_2],
            max_iter=50000,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
    
    def neural_net(self, x: torch.Tensor, y: torch.Tensor, 
                   t: torch.Tensor) -> torch.Tensor:
        X = torch.cat([x, y, t], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(H)
    
    def net_NS(self, x: torch.Tensor, y: torch.Tensor, 
               t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = self.neural_net(x, y, t)
        psi = output[:, 0:1]  # Stream function
        p = output[:, 1:2]    # Pressure
        
        # Velocities from stream function: u = ∂ψ/∂y, v = -∂ψ/∂x
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), 
                                create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), 
                                 create_graph=True)[0]
        
        # Compute derivatives for u
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                                  create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                  create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                                  create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                   create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), 
                                   create_graph=True)[0]
        
        # Compute derivatives for v
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), 
                                  create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), 
                                  create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), 
                                  create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), 
                                   create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), 
                                   create_graph=True)[0]
        
        # Pressure gradients
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), 
                                  create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), 
                                  create_graph=True)[0]
        
        # Navier-Stokes residuals
        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy)
        
        return u, v, p, f_u, f_v
    
    def compute_loss(self, iteration: int) -> Tuple[torch.Tensor, ...]:
        # 1. Initial condition loss (t=0)
        u_pred_ic, v_pred_ic, _, _, _ = self.net_NS(self.x_ic, self.y_ic, self.t_ic)
        loss_ic_u = torch.mean((self.u_ic - u_pred_ic) ** 2)
        loss_ic_v = torch.mean((self.v_ic - v_pred_ic) ** 2)
        loss_ic = loss_ic_u + loss_ic_v
        
        # 2. Boundary condition loss
        u_pred_bc, v_pred_bc, _, _, _ = self.net_NS(self.x_bc, self.y_bc, self.t_bc)
        loss_bc_u = torch.mean((self.u_bc - u_pred_bc) ** 2)
        loss_bc_v = torch.mean((self.v_bc - v_pred_bc) ** 2)
        loss_bc = loss_bc_u + loss_bc_v
        
        # 3. Physics residual loss
        _, _, _, f_u_pred, f_v_pred = self.net_NS(
            self.x_unlabeled, self.y_unlabeled, self.t_unlabeled
        )
        loss_physics_u = torch.mean(f_u_pred ** 2)
        loss_physics_v = torch.mean(f_v_pred ** 2)
        loss_physics = loss_physics_u + loss_physics_v
        
        # Total weighted loss
        loss_total = (self.alpha_ic * loss_ic + 
                     self.alpha_bc * loss_bc + 
                     self.alpha_physics * loss_physics)
        
        # Store for tracking
        self.loss_history['iteration'].append(iteration)
        self.loss_history['total'].append(loss_total.item())
        self.loss_history['ic'].append(loss_ic.item())
        self.loss_history['bc'].append(loss_bc.item())
        self.loss_history['physics'].append(loss_physics.item())
        self.loss_history['lambda_1'].append(self.lambda_1.item())
        self.loss_history['lambda_2'].append(self.lambda_2.item())
        
        return loss_total, loss_ic, loss_bc, loss_physics
    
    def train_adam(self, nIter: int):
        self.model.train()
        start_time = time.time()
        
        for it in range(nIter):
            self.optimizer.zero_grad()
            loss_total, loss_ic, loss_bc, loss_physics = self.compute_loss(it)
            loss_total.backward()
            self.optimizer.step()
            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                print(f'It: {it:5d} | '
                      f'Total: {loss_total.item():.3e} | '
                      f'IC: {loss_ic.item():.3e} | '
                      f'BC: {loss_bc.item():.3e} | '
                      f'Phy: {loss_physics.item():.3e} | '
                      f'λ1: {self.lambda_1.item():.5f} | '
                      f'λ2: {self.lambda_2.item():.5f} | '
                      f'Time: {elapsed:.2f}s')
                start_time = time.time()
    
    def train_lbfgs(self):
        self.model.train()
        iteration_counter = [len(self.loss_history['iteration'])]
        
        def closure():
            self.optimizer_lbfgs.zero_grad()
            loss_total, _, _, _ = self.compute_loss(iteration_counter[0])
            loss_total.backward()
            iteration_counter[0] += 1
            return loss_total
        
        self.optimizer_lbfgs.step(closure)
        
        loss_total, loss_ic, loss_bc, loss_physics = self.compute_loss(iteration_counter[0])
        print(f'\nLBFGS Final - Total: {loss_total.item():.3e} | '
              f'IC: {loss_ic.item():.3e} | '
              f'BC: {loss_bc.item():.3e} | '
              f'Phy: {loss_physics.item():.3e} | '
              f'λ1: {self.lambda_1.item():.5f} | '
              f'λ2: {self.lambda_2.item():.5f}')
    
    def train(self, nIter_adam: int = 100000, use_lbfgs: bool = True):
        print("\n" + "="*80)
        print("PINN TRAINING")
        print("="*80)
        print(f"Loss weights: α_IC={self.alpha_ic}, α_BC={self.alpha_bc}, "
              f"α_Physics={self.alpha_physics}")
        
        print("\nTraining with Adam optimizer...")
        self.train_adam(nIter_adam)
        
        if use_lbfgs:
            print("\nFine-tuning with L-BFGS optimizer...")
            self.train_lbfgs()
        else:
            print("\nSkipping L-BFGS optimization")
    
    def predict(self, x_star: np.ndarray, y_star: np.ndarray, 
                t_star: np.ndarray, batch_size: int = 5000) -> Tuple[np.ndarray, ...]:
        self.model.eval()
        
        n_samples = x_star.shape[0]
        u_pred = np.zeros((n_samples, 1))
        v_pred = np.zeros((n_samples, 1))
        p_pred = np.zeros((n_samples, 1))
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                
                x_batch = torch.tensor(x_star[i:end_idx], dtype=torch.float32, 
                                      requires_grad=True).to(self.device)
                y_batch = torch.tensor(y_star[i:end_idx], dtype=torch.float32, 
                                      requires_grad=True).to(self.device)
                t_batch = torch.tensor(t_star[i:end_idx], dtype=torch.float32, 
                                      requires_grad=True).to(self.device)
                
                u_batch, v_batch, p_batch, _, _ = self.net_NS(x_batch, y_batch, t_batch)
                
                u_pred[i:end_idx] = u_batch.detach().cpu().numpy()
                v_pred[i:end_idx] = v_batch.detach().cpu().numpy()
                p_pred[i:end_idx] = p_batch.detach().cpu().numpy()
                
                del x_batch, y_batch, t_batch, u_batch, v_batch, p_batch
                torch.cuda.empty_cache()
        
        return u_pred, v_pred, p_pred
    
    def save_model(self, output_dir: str = 'results', 
                   timestamp: Optional[str] = None) -> str:
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'lambda_1': self.lambda_1.item(),
            'lambda_2': self.lambda_2.item(),
            'lb': self.lb.cpu().numpy(),
            'ub': self.ub.cpu().numpy(),
            'loss_history': self.loss_history,
            'alpha_ic': self.alpha_ic,
            'alpha_bc': self.alpha_bc,
            'alpha_physics': self.alpha_physics
        }
        
        model_path = os.path.join(output_dir, f'model_checkpoint_{timestamp}.pth')
        torch.save(checkpoint, model_path)
        print(f"\nSaved model checkpoint to: {model_path}")
        
        return model_path
    
    def plot_visualizations(self, X_star: np.ndarray, t_star: np.ndarray,
                           u_pred: np.ndarray, v_pred: np.ndarray, p_pred: np.ndarray,
                           p_star_snap: np.ndarray, output_dir: str = 'results',
                           timestamp: Optional[str] = None):
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        os.makedirs(output_dir, exist_ok=True)
        
        x_star = X_star[:, 0:1]
        y_star = X_star[:, 1:2]
        
        # Interpolate for plotting
        print("\nInterpolating for plotting...")
        nn = 200
        x_plot = np.linspace(x_star.min(), x_star.max(), nn)
        y_plot = np.linspace(y_star.min(), y_star.max(), nn)
        X, Y = np.meshgrid(x_plot, y_plot)
        
        UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
        VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
        PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
        P_exact = griddata(X_star, p_star_snap.flatten(), (X, Y), method='cubic')
        
        # ========================================================================
        # PLOT 1: 3D CONTOUR PLOT
        # ========================================================================
        
        print("Plotting 3D contour visualization...")
        
        fig = plt.figure(figsize=(14, 6))
        
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(top=0.95, bottom=0.05, left=0.01, right=0.99, wspace=0.05)
        
        # Define plot ranges
        r1 = [x_star.min(), x_star.max()]
        r2 = [t_star.min(), t_star.max()]
        r3 = [y_star.min(), y_star.max()]
        
        # u(t,x,y)
        ax = plt.subplot(gs1[:, 0], projection='3d')
        ax.axis('off')
        
        # Draw bounding box
        for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
            if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
                ax.plot3D(*zip(s, e), color="k", linewidth=0.8)
        
        # Plot velocity contour at MIDDLE of time axis
        ax.contourf(X, UU_star, Y, zdir='y', offset=t_star.mean(), cmap='rainbow', alpha=0.9, levels=30)
        
        # Add text labels
        ax.text(x_star.mean(), r2[0] - 1.5, r3[0] - 1, '$x$', fontsize=12)
        ax.text(r1[1]+1, t_star.mean(), r3[0] - 1, '$t$', fontsize=12)
        ax.text(r1[0]-1.5, r2[0] - 0.5, y_star.mean(), '$y$', fontsize=12)
        ax.text(r1[0]-3, t_star.mean(), r3[1] + 1, '$u(t,x,y)$', fontsize=14, weight='bold')
        ax.set_xlim3d(r1)
        ax.set_ylim3d(r2)
        ax.set_zlim3d(r3)
        axisEqual3D(ax)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=-60)
        
        # v(t,x,y)
        ax = plt.subplot(gs1[:, 1], projection='3d')
        ax.axis('off')
        
        # Draw bounding box
        for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
            if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
                ax.plot3D(*zip(s, e), color="k", linewidth=0.8)
        
        # Plot velocity contour at MIDDLE of time axis
        ax.contourf(X, VV_star, Y, zdir='y', offset=t_star.mean(), cmap='rainbow', alpha=0.9, levels=30)
        
        # Add text labels
        ax.text(x_star.mean(), r2[0] - 1.5, r3[0] - 1, '$x$', fontsize=12)
        ax.text(r1[1]+1, t_star.mean(), r3[0] - 1, '$t$', fontsize=12)
        ax.text(r1[0]-1.5, r2[0] - 0.5, y_star.mean(), '$y$', fontsize=12)
        ax.text(r1[0]-3, t_star.mean(), r3[1] + 1, '$v(t,x,y)$', fontsize=14, weight='bold')
        ax.set_xlim3d(r1)
        ax.set_ylim3d(r2)
        ax.set_zlim3d(r3)
        axisEqual3D(ax)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'3d_contour_{timestamp}.svg'), format='svg', bbox_inches='tight')
        print(f"Saved: 3d_contour_{timestamp}.svg")
        plt.close()
        
        # ========================================================================
        # PLOT 2: PRESSURE COMPARISON
        # ========================================================================
        
        print("Plotting pressure comparison...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        im1 = axes[0].imshow(PP_star, interpolation='nearest', cmap='rainbow',
                            extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                            origin='lower', aspect='auto')
        axes[0].set_xlabel('$x$')
        axes[0].set_ylabel('$y$')
        axes[0].set_title('Predicted Pressure')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(P_exact, interpolation='nearest', cmap='rainbow',
                            extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                            origin='lower', aspect='auto')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$y$')
        axes[1].set_title('Exact Pressure')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pressure_comparison_{timestamp}.svg'), format='svg', bbox_inches='tight')
        print(f"Saved: pressure_comparison_{timestamp}.svg")
        plt.close()
        
        # ========================================================================
        # PLOT 3: CONVERGENCE HISTORY
        # ========================================================================
        
        print("Plotting convergence history...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Total loss
        axes[0, 0].semilogy(self.loss_history['total'], 'k-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss components
        axes[0, 1].semilogy(self.loss_history['ic'], 'r-', linewidth=2, label='IC')
        axes[0, 1].semilogy(self.loss_history['bc'], 'g-', linewidth=2, label='BC')
        axes[0, 1].semilogy(self.loss_history['physics'], 'm-', linewidth=2, label='Physics')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IC vs BC loss
        axes[0, 2].semilogy(self.loss_history['ic'], 'r-', linewidth=2, label=f'IC (α={self.alpha_ic})')
        axes[0, 2].semilogy(self.loss_history['bc'], 'g-', linewidth=2, label=f'BC (α={self.alpha_bc})')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('IC vs BC Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Lambda 1
        axes[1, 0].plot(self.loss_history['lambda_1'], 'b-', linewidth=2, label='λ₁')
        axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target (1.0)')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('λ₁')
        axes[1, 0].set_title('Lambda 1 Convergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Lambda 2
        axes[1, 1].plot(self.loss_history['lambda_2'], 'r-', linewidth=2, label='λ₂')
        axes[1, 1].axhline(y=0.01, color='b', linestyle='--', alpha=0.5, label='Target (0.01)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('λ₂')
        axes[1, 1].set_title('Lambda 2 Convergence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 2].axis('off')
        lambda_1_value = self.lambda_1.item()
        lambda_2_value = self.lambda_2.item()
        summary_text = f"""
        TRAINING SUMMARY
        ════════════════════════════════
        
        Loss Weights:
        α_IC:             {self.alpha_ic}
        α_BC:             {self.alpha_bc}
        α_Physics:        {self.alpha_physics}
        
        Data Points:
        IC points:        {len(self.x_ic)}
        BC points:        {len(self.x_bc)}
        Unlabeled points: {len(self.x_unlabeled)}
        
        Final Results:
        λ₁:               {lambda_1_value:.5f}
        λ₂:               {lambda_2_value:.5f}
        """
        axes[1, 2].text(0.5, 0.5, summary_text, fontsize=11, ha='center', va='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'convergence_history_{timestamp}.svg'), format='svg', bbox_inches='tight')
        print(f"Saved: convergence_history_{timestamp}.svg")
        plt.close()


def identify_boundary_points(X_star: np.ndarray, 
                            tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    x = X_star[:, 0]
    y = X_star[:, 1]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Points on boundaries
    boundary_mask = (
        (np.abs(x - x_min) < tolerance) |  # left boundary
        (np.abs(x - x_max) < tolerance) |  # right boundary
        (np.abs(y - y_min) < tolerance) |  # bottom boundary
        (np.abs(y - y_max) < tolerance)    # top boundary
    )
    
    boundary_indices = np.where(boundary_mask)[0]
    interior_indices = np.where(~boundary_mask)[0]
    
    return boundary_indices, interior_indices


def prepare_data_icbc_only(X_star: np.ndarray, U_star: np.ndarray, 
                           t_star: np.ndarray, 
                           n_unlabeled: int) -> Tuple[np.ndarray, ...]:
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Identify boundary and interior points
    boundary_indices, interior_indices = identify_boundary_points(X_star)
    
    print(f"\nSpatial domain analysis:")
    print(f"  Total spatial points: {N}")
    print(f"  Boundary points: {len(boundary_indices)}")
    print(f"  Interior points: {len(interior_indices)}")
    
    # Create meshgrid
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(t_star, (1, N)).T
    
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    
    # Initial condition (t=0, all spatial points)
    ic_time_idx = 0
    ic_spatial_indices = np.arange(N)
    ic_indices = ic_spatial_indices + ic_time_idx * N
    
    x_ic = XX.flatten()[ic_indices, None]
    y_ic = YY.flatten()[ic_indices, None]
    t_ic = TT.flatten()[ic_indices, None]
    u_ic = UU.flatten()[ic_indices, None]
    v_ic = VV.flatten()[ic_indices, None]
    
    # Boundary condition (boundary at all t)
    bc_indices = []
    for t_idx in range(T):
        for spatial_idx in boundary_indices:
            bc_indices.append(spatial_idx + t_idx * N)
    bc_indices = np.array(bc_indices)
    
    x_bc = XX.flatten()[bc_indices, None]
    y_bc = YY.flatten()[bc_indices, None]
    t_bc = TT.flatten()[bc_indices, None]
    u_bc = UU.flatten()[bc_indices, None]
    v_bc = VV.flatten()[bc_indices, None]
    
    # Unlabeled data (random interior, excluding IC/BC)
    all_indices = np.arange(N * T)
    supervised_indices = np.concatenate([ic_indices, bc_indices])
    unlabeled_pool = np.setdiff1d(all_indices, supervised_indices)
    
    n_unlabeled_actual = min(n_unlabeled, len(unlabeled_pool))
    unlabeled_indices = np.random.choice(unlabeled_pool, n_unlabeled_actual, replace=False)
    
    x_unlabeled = XX.flatten()[unlabeled_indices, None]
    y_unlabeled = YY.flatten()[unlabeled_indices, None]
    t_unlabeled = TT.flatten()[unlabeled_indices, None]
    
    print(f"\nData preparation summary:")
    print(f"  IC points (t=0, all spatial): {len(ic_indices)}")
    print(f"  BC points (boundary, all t): {len(bc_indices)}")
    print(f"  Unlabeled points: {len(unlabeled_indices)}")
    print(f"  Total supervised: {len(ic_indices) + len(bc_indices)}")
    print(f"  Total points: {len(ic_indices) + len(bc_indices) + len(unlabeled_indices)}")
    
    return (x_ic, y_ic, t_ic, u_ic, v_ic,
            x_bc, y_bc, t_bc, u_bc, v_bc,
            x_unlabeled, y_unlabeled, t_unlabeled)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)


def main(data_path: str, output_dir: str = 'results', 
         n_unlabeled: int = 10000, nIter_adam: int = 100000, 
         use_lbfgs: bool = True):
    print("="*80)
    print("PINN FOR NAVIER-STOKES EQUATIONS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = scipy.io.loadmat(data_path)
    
    X_star = data['X_star']  # N x 2
    U_star = data['U_star']  # N x 2 x T
    t_star = data['t']       # T x 1
    P_star = data['p_star']  # N x T
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    print(f"Data loaded:")
    print(f"  Spatial points: {N}")
    print(f"  Time steps: {T}")
    print(f"  Total spatiotemporal points: {N * T}")
    
    # Prepare data
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    data_tuple = prepare_data_icbc_only(X_star, U_star, t_star, n_unlabeled)
    
    # Network architecture
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    print(f"\nNetwork architecture: {layers}")
    
    # Train model
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    model = PINN_ICBC(
        *data_tuple, 
        layers, 
        alpha_ic=1.0,
        alpha_bc=1.0,
        alpha_physics=1.0
    )
    
    model.train(nIter_adam=nIter_adam, use_lbfgs=use_lbfgs)
    
    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save_model(output_dir, timestamp)
    
    # Final results
    lambda_1_value = model.lambda_1.item()
    lambda_2_value = model.lambda_2.item()
    
    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - 0.01) / 0.01 * 100
    
    print('\n' + '='*80)
    print('FINAL RESULTS')
    print('='*80)
    print(f'λ₁: {lambda_1_value:.5f} (target: 1.0, error: {error_lambda_1:.2f}%)')
    print(f'λ₂: {lambda_2_value:.5f} (target: 0.01, error: {error_lambda_2:.2f}%)')
    
    # Prediction and error analysis
    print("\n" + "="*80)
    print("PREDICTION AND ERROR ANALYSIS")
    print("="*80)
    
    snap = np.array([100])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(t_star, (1, N)).T
    
    t_snap = TT[:, snap]
    u_star_snap = U_star[:, 0, snap]
    v_star_snap = U_star[:, 1, snap]
    p_star_snap = P_star[:, snap]
    
    print(f"Predicting on snapshot {snap[0]}...")
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_snap)
    
    # Compute errors
    error_u = np.linalg.norm(u_star_snap - u_pred, 2) / np.linalg.norm(u_star_snap, 2)
    error_v = np.linalg.norm(v_star_snap - v_pred, 2) / np.linalg.norm(v_star_snap, 2)
    error_p = np.linalg.norm(p_star_snap - p_pred, 2) / np.linalg.norm(p_star_snap, 2)
    
    print(f"\nRelative L2 errors:")
    print(f"  u-velocity: {error_u*100:.2f}%")
    print(f"  v-velocity: {error_v*100:.2f}%")
    print(f"  Pressure: {error_p*100:.2f}%")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    model.plot_visualizations(X_star, t_star, u_pred, v_pred, p_pred, 
                             p_star_snap, output_dir, timestamp)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"Files generated:")
    print(f"  1. model_checkpoint_{timestamp}.pth")
    print(f"  2. 3d_contour_{timestamp}.svg")
    print(f"  3. pressure_comparison_{timestamp}.svg")
    print(f"  4. convergence_history_{timestamp}.svg")


if __name__ == '__main__':
    # Configuration
    DATA_PATH = 'ellip_cylinder_wake.mat'  # Update this path
    OUTPUT_DIR = 'results'
    N_UNLABELED = 10000
    N_ITER_ADAM = 100000
    USE_LBFGS = True
    
    # Run training
    main(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        n_unlabeled=N_UNLABELED,
        nIter_adam=N_ITER_ADAM,
        use_lbfgs=USE_LBFGS
    )