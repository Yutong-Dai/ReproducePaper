'''
File: TrustRegionCG.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-04-13 22:25
Last Modified: 2021-04-14 21:51
--------------------------------------------
Description:
'''
import torch
import utils
import numpy as np


class TrusRegionCG:
    def __init__(self, x, radius_init=0.1, radius_max=10, eta=0.2):
        """
            x (list): a list of parameters; model.parameters()
        """
        self.x = x
        self.device = x[0].device.type
        self.radius = radius_init
        self.radius_max = radius_max
        self.eta = eta

        self.cgmaxiter = 0
        for e in x:
            self.cgmaxiter += len(e.view(-1))

    def _steihaug(self, gradf, radius):
        """
            reference: P171. Numerical Optization (Stephan & Jorge) 2ed;
        """
        self.cg_iter = 0
        z = []
        for e in self.x:
            z.append(torch.zeros_like(e))
        r = []
        d = []

        # calculate the norm of the gradient at the starting point
        norm_gradf0 = 0.0
        for e in gradf:
            # +0.0 to do a copy
            r.append(e.data + 0.0)
            d.append(0.0 - e.data)
            norm_gradf0 += torch.norm(e.data)**2
        norm_gradf0 = (norm_gradf0.data.item()) ** 0.5
        self.norm_gradf0 = norm_gradf0
        cg_tol = min(0.5, norm_gradf0**0.5) * norm_gradf0
        if norm_gradf0 < cg_tol:
            self.cgflag = 'cgtol'
            return z
        while True:
            self.cg_iter += 1
            # check termination
            if self.cg_iter > self.cgmaxiter:
                print("Reach cg max iterations!")
                d = z
                self.cgflag = 'cgmax'
                break
            # hessian vector product
            Hd = torch.autograd.grad(gradf, self.x, d, retain_graph=True)
            # negative curvature test
            dtHd = 0.0
            for idx, hd in enumerate(Hd):
                dtHd += (hd * d[idx]).sum()
            if dtHd.data.item() <= 0.0:
                tau = self._findroots(z, d, radius)
                for idx in range(len(self.x)):
                    d[idx] = z[idx] + tau * d[idx]
                self.cgflag = 'negcv'
                break
            # positive curvature

            norm_r_sq = 0.0
            for e in r:
                norm_r_sq += (e * e).sum()
            alpha = (norm_r_sq / dtHd).data.item()

            znew = []
            norm_znew = 0.0
            for idx in range(len(self.x)):
                trial = z[idx] + alpha * d[idx] + 0.0
                znew.append(trial)
                norm_znew += torch.norm(trial)**2
            norm_znew = (norm_znew ** 0.5).data.item()

            if norm_znew >= radius:
                tau = self._findroots(z, d, radius)
                for idx in range(len(self.x)):
                    d[idx] = z[idx] + tau * d[idx] + 0.0
                self.cgflag = 'posbd'
                break
            rnew = []
            norm_rnew = 0.0
            for idx in range(len(self.x)):
                temp = r[idx] + alpha * Hd[idx] + 0.0
                rnew.append(temp)
                norm_rnew += torch.norm(temp)**2
            norm_rnew = norm_rnew**0.5.data.item()
            if norm_rnew < cg_tol:
                d = znew
                self.cgflag = 'cgtol'
            beta = (norm_rnew**2 / norm_r_sq).data.item()
            for idx in range(len(self.x)):
                d[idx] = -rnew[idx] + beta * d[idx]
        return d

    def _findroots(self, z, d, radius):
        a, b, c = 0.0, 0.0, 0.0
        for idx in range(len(z)):
            a += (d[idx] * d[idx]).sum()
            b += (d[idx] * z[idx]).sum()
            c += (z[idx] * d[idx]).sum()
        b *= 2.0
        c -= radius**2
        tau = (-2.0 * c) / (b + (b**2 - (4.0 * a * c))**0.5)
        return tau.data.item()

    def step(self, loss_fn, aCs, aSs, layers, style_layer_weights,
             c_layer, alpha, beta):
        """
            customized step
            loss_fn: callable
            aCs, aSs, style_layer_weights, c_layer, alpha, beta: function parameters
        """
        aGs = utils.get_feature_maps(self.x[0], layers)
        loss, content_cost, style_cost = loss_fn(aGs, aCs, aSs, style_layer_weights,
                                                 content_layer_idx=c_layer, alpha=alpha, beta=beta)
        print(f'loss:{loss.data.cpu().item():2.3e} | content: {content_cost.item():2.3e} | style_cost:{style_cost.item():2.3e}', flush=True)
        gradf = torch.autograd.grad(loss, self.x, create_graph=True)
        p = self._steihaug(gradf, self.radius)
        print(f'   CG-Steihaug: current gradf_norm:{self.norm_gradf0:3.3e} | {self.cg_iter}/{self.cgmaxiter} | terminate with: {self.cgflag}')
        # actual decrease at the trial point
        with torch.no_grad():
            xtrial = []
            for idx in range(len(self.x)):
                xtrial.append(self.x[idx] + p[idx] + 0.0)
        aGnews = utils.get_feature_maps(xtrial[0], layers)
        with torch.no_grad():
            loss_new, _, _ = loss_fn(aGnews, aCs, aSs, style_layer_weights,
                                     content_layer_idx=c_layer, alpha=alpha, beta=beta)
        actual_decrease = loss - loss_new
        # model decrease at the trial point
        Hp = torch.autograd.grad(gradf, self.x, p)
        ptHp = 0.0
        for idx, hp in enumerate(Hp):
            ptHp += (hp * p[idx]).sum()
        gp = 0.0
        for idx, e in enumerate(gradf):
            gp += (e.data * p[idx]).sum()
        model_decrease = -gp.data.item() - (ptHp.data.item()) / 2
        rho = actual_decrease / model_decrease
        norm_p = 0.0
        for e in p:
            norm_p += torch.norm(e)**2
        norm_p = (norm_p ** 0.5).data.item()
        if rho < 1 / 4:
            self.radius *= 0.25
            radius_flag = 'shrink'
        else:
            if rho > 3 / 4 and np.abs(norm_p - self.radius) <= 1e-10:
                self.radius = min(2 * self.radius, self.radius_max)
                radius_flag = 'enlarge'
            else:
                radius_flag = 'unchanged'
        if rho > self.eta:
            for idx, e in enumerate(self.x):
                e.data = e.data + p[idx].data
                update_flag = 'move'
        else:
            update_flag = 'stay'
        print(f'   Trust-Region: {radius_flag:10s} | new radius:{self.radius:3.3e} | x-update:{update_flag}')

    def zero_grad(self):
        """
            just 
        """
        pass
