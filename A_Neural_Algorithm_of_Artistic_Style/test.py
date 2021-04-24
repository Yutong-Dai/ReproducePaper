class TRCGOptimizer:
    def __init__(self,x,device,radius,cgopttol=1e-7,c0tr=0.2,c1tr=0.25,c2tr=0.75,t1tr=0.25,t2tr=2.0,radius_max=5.0,\
                 radius_initial=0.1):
        
        self.x = x
        self.device = device
        self.cgopttol = cgopttol
        self.c0tr = c0tr
        self.c1tr = c1tr
        self.c2tr = c2tr
        self.t1tr = t1tr
        self.t2tr = t2tr
        self.radius_max = radius_max
        self.radius_initial = radius_initial
        self.radius = radius
        self.cgmaxiter = 60
        
    def findroot(self,x,p):
        
        aa = 0.0
        bb = 0.0
        cc = 0.0
    
        for e in range(len(x)):
            aa += (p[e]*p[e]).sum()
            bb += (p[e]*x[e]).sum()
            cc += (x[e]*x[e]).sum()
        
        bb = bb*2.0
        cc = cc - self.radius**2
    
        alpha = (-2.0*cc)/(bb+(bb**2-(4.0*aa*cc)).sqrt())

        return alpha.data.item()
    
    
    def CGSolver(self,loss_grad,cnt_compute):
    
        cg_iter = 0 # iteration counter
        x0 = [] # define x_0 as a list
        for i in self.x:
            x0.append(torch.zeros(i.shape).to(self.device))
    
        r0 = [] # set initial residual to gradient
        p0 = [] # set initial conjugate direction to -r0
        self.cgopttol = 0.0
        
        for i in loss_grad:
            r0.append(i.data+0.0)     
            p0.append(0.0-i.data)
            self.cgopttol+=torch.norm(i.data)**2
        
        self.cgopttol = self.cgopttol.data.item()**0.5
        self.cgopttol = (min(0.5,self.cgopttol**0.5))*self.cgopttol
    
        cg_term = 0
        j = 0

        while 1:
            j+=1
    
            # if CG does not solve model within max allowable iterations
            if j > self.cgmaxiter:
                j=j-1
                p1 = x0
                print ('\n\nCG has issues !!!\n\n')
                break
            # hessian vector product
            loss_grad_direct = 0.0
            ind = 0
            for i in loss_grad:
                loss_grad_direct += (i*p0[ind]).sum()
                ind+=1
            Hp = torch.autograd.grad(loss_grad_direct,self.x,retain_graph=True) # hessian-vector in tuple
            cnt_compute+=1
            
            pHp = 0.0 # quadratic term
            ind = 0
            for i in Hp:
                pHp += (p0[ind]*i).sum()
                ind+=1
    
            # if nonpositive curvature detected, go for the boundary of trust region
            if pHp.data.item() <= 0:
                tau = self.findroot(x0,p0)
                p1 = []
                for e in range(len(x0)):
                    p1.append(x0[e]+tau*p0[e])
                cg_term = 1
                break
            
            # if positive curvature
            # vector product
            rr0 = 0.0
            for i in r0:
                rr0 += (i*i).sum()
            
            # update alpha
            alpha = (rr0/pHp).data.item()
        
            x1 = []
            norm_x1 = 0.0
            for e in range(len(x0)):
                x1.append(x0[e]+alpha*p0[e])
                norm_x1 += torch.norm(x0[e]+alpha*p0[e])**2
            norm_x1 = norm_x1**0.5
            
            # if norm of the updated x1 > radius
            if norm_x1.data.item() >= self.radius:
                tau = self.findroot(x0,p0)
                p1 = []
                for e in range(len(x0)):
                    p1.append(x0[e]+tau*p0[e])
                cg_term = 2
                break
    
            # update residual
            r1 = []
            norm_r1 = 0.0
            for e in range(len(r0)):
                r1.append(r0[e]+alpha*Hp[e])
                norm_r1 += torch.norm(r0[e]+alpha*Hp[e])**2
            norm_r1 = norm_r1**0.5
    
            if norm_r1.data.item() < self.cgopttol:
                p1 = x1
                cg_term = 3
                break
    
            rr1 = 0.0
            for i in r1:
                rr1 += (i*i).sum()
    
            beta = (rr1/rr0).data.item()
    
            # update conjugate direction for next iterate
            p1 = []
            for e in range(len(r1)):
                p1.append(-r1[e]+beta*p0[e])
    
            p0 = p1
            x0 = x1
            r0 = r1
    

        cg_iter = j
        d = p1

        return d,cg_iter,cg_term,cnt_compute
    
    def step(self, loss_fn, aCs, aSs, layers, style_layer_weights,
             c_layer, alpha, beta):
        
        update = 2
        aGs = utils.get_feature_maps(self.x[0], layers)
        loss, content_cost, style_cost = loss_fn(aGs, aCs, aSs, style_layer_weights,
                                                 content_layer_idx=c_layer, alpha=alpha, beta=beta)
        print(f'loss:{loss.data.cpu().item():2.3e} | content: {content_cost.item():2.3e} | style_cost:{style_cost.item():2.3e}', flush=True)
        loss_grad = torch.autograd.grad(loss, self.x, create_graph=True)
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
    
        loss_grad = torch.autograd.grad(loss,self.x,create_graph=True) 
        NormG = torch.sqrt(np.sum([(li.data*li.data).sum() for li in loss_grad])).data.item()
        ListG = [torch.sqrt((li.data*li.data).sum()) for li in loss_grad]
    
        cnt_compute=1
        
        # Conjugate Gradient Method
        d, cg_iter, cg_term, cnt_compute = self.CGSolver(loss_grad,cnt_compute)

        # current iterate 
        w0 = []
        for i in self.x:
            w0.append(i.data.cpu().numpy())
    
        # update solution
        ind = 0
        for i in self.x:
            i.data = torch.from_numpy(w0[ind]).to(self.device)+d[ind]
            ind+=1
    
        # MSE loss plus penalty term
        with torch.no_grad():
            loss_new = MSE(Pred_new,y_time_series)

        numerator = loss.data.item() - loss_new.data.item()

        # dHd
        Hd = 0.0
        loss_grad_direct = 0.0
        ind = 0
        for i in loss_grad:
            loss_grad_direct += (i*d[ind]).sum()
            ind+=1
        Hd = torch.autograd.grad(loss_grad_direct,self.x) # hessian-vector in tuple
        dHd = 0.0 # quadratic term
        ind = 0
        for i in Hd:
            dHd += (d[ind]*i.data).sum()
            ind+=1

        gd = 0.0
        ind = 0
        for i in loss_grad:
            gd += (i.data*d[ind]).sum()
            ind+=1

        norm_d = 0.0
        for i in d:
            norm_d += torch.norm(i)**2
        norm_d = norm_d**0.5
        
        denominator = -gd.data.item() - 0.5*(dHd.data.item())

        # ratio
        rho = numerator/denominator

        if rho < self.c1tr: # shrink radius
            self.radius = self.t1tr*self.radius
            update = 0
        if rho > self.c2tr and np.abs(norm_d.data.item() - self.radius) < 1e-10: # enlarge radius
            self.radius = min(self.t2tr*self.radius,self.radius_max)
            update = 1
        # otherwise, radius remains the same
        
        if rho <= self.c0tr: # reject d
            update = 3
            ind = 0
            for i in self.x:
                i.data = torch.from_numpy(w0[ind]).to(self.device)
                ind+=1
    
        # return d, rho, update, cg_iter, cg_term, loss_grad, norm_d, numerator, denominator, w0
        return self.radius, cnt_compute, cg_iter

optimizer = TRCGOptimizer([G], "cuda", radius_initial)