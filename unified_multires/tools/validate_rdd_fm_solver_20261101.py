"""CPU algebra audit and analytic-field benchmark; no weights/data downloads.

Run using activate_cpu.sh and the project's UMDIFF_PYTHON, with --output.
Errors are ODE trajectory RMSE, NOT FID, pixel quality or native model parity.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.integrate import solve_ivp, quad
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'unified_multires'))
from research.rdd_fm_integrator import Stage
from umdiff.paths import RDDFlow, EndpointFlow


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def algebra():
    old = module('audit_image_fm', 'multiscale_diffusion/diffusion/flow_matching_with_drift.py')
    video = module('audit_video_fm', 'multiscale_diffusion/rdd_video/flow_matching_video.py')
    native = module('audit_fv_flow', 'FastVideo-RDD-minimal/rdd/flow.py').RDDPath()
    torch.manual_seed(481)
    rows = []
    for i, (a, b) in enumerate(zip(native.boundaries, native.boundaries[1:])):
        x = torch.randn(5, 2, 8, 8, dtype=torch.float64)
        eps = torch.randn_like(x)
        t = torch.linspace(a, b, len(x), dtype=torch.float64)
        fm = old.FlowMatchingWithDrift(c=None if i == 0 else i, s_k=a, s_k_plus_1=b)
        unified = RDDFlow(start=a, end=b, drift=i > 0)
        d = None if i == 0 else fm.compute_drift(x)
        z = fm.q_sample(x, t, eps, d)
        v = fm.compute_velocity_target(x, eps, t, d)
        uz = unified.sample(x, t, eps)
        y = fm.predict_x1_from_velocity(z, v, t)
        nvz, nvw = native.noisy_and_target(x.unsqueeze(2), eps.unsqueeze(2), t, i)
        recovered = native.clean_from_velocity(nvz, nvw, t, i).squeeze(2)
        vf = video.VideoFlowMatchingWithDrift(c=None if i == 0 else i,
                    spatial_factor=2, temporal_factor=1, s_k=a, s_k_plus_1=b)
        vd = None if i == 0 else vf.compute_drift(x.unsqueeze(2))
        vv = vf.compute_velocity_target(x.unsqueeze(2), eps.unsqueeze(2), t, vd)
        vy = vf.predict_x1_from_velocity(nvz, vv, 1 - t).squeeze(2)
        dt = 1e-6
        fd = (fm.q_sample(x, t + dt, eps, d) - fm.q_sample(x, t - dt, eps, d)) / (2 * dt)
        checks = {'unified_state': (uz.state-z).abs().max().item(),
                  'unified_velocity': (uz.velocity-v).abs().max().item(),
                  'legacy_clean': (y-x).abs().max().item(),
                  'unified_clean': (unified.clean_from_velocity(z,v,t)-x).abs().max().item(),
                  'fastvideo_state': (nvz.squeeze(2)-z).abs().max().item(),
                  'wan_velocity_sign': (nvw.squeeze(2)+v).abs().max().item(),
                  'fastvideo_clean': (recovered-x).abs().max().item(),
                  'video_clean': (vy-x).abs().max().item(),
                  'finite_difference_velocity': (fd-v).abs().max().item()}
        assert max(checks.values()) < 2e-8, checks
        rows.append({'stage': [a,b], 'max_abs_errors': checks})
    temporal_rows=[]
    temporal_path=type(native)(factors=((4,4),(2,2),(1,1)))
    full=torch.randn(4,2,8,16,16,dtype=torch.float64)
    for i,(a,b) in enumerate(zip(native.boundaries,native.boundaries[1:])):
        x=temporal_path.stage_clean(full,i);eps=torch.randn_like(x)
        t=torch.linspace(a,b,len(x),dtype=x.dtype)
        z,w=temporal_path.noisy_and_target(x,eps,t,i)
        vf=video.VideoFlowMatchingWithDrift(c=None if i==0 else i,
                 spatial_factor=2,temporal_factor=2,s_k=a,s_k_plus_1=b)
        d=None if i==0 else vf.compute_drift(x)
        vv=vf.compute_velocity_target(x,eps,t,d)
        vz=vf.q_sample(x,t,eps,d)
        err=max(float((vv-w).abs().max()),float((vz-z).abs().max()),
                float((vf.predict_x1_from_velocity(z,w,1-t)-x).abs().max()),
                float((temporal_path.clean_from_velocity(z,w,t,i)-x).abs().max()))
        assert err<1e-12
        temporal_rows.append({'stage':[a,b],'shape':list(x.shape),'max_abs_error':err})
    # A non-idempotent configured operator cannot use the projection inverse.
    bad = RDDFlow(start=.4,end=.7,projection={'down':'average','up':'bilinear'})
    x = torch.randn(3,2,8,8,dtype=torch.float64);eps=torch.randn_like(x)
    t = torch.full((3,),.5,dtype=torch.float64);out=bad.sample(x,t,eps)
    try:
        wrong=bad.clean_from_velocity(out.state,out.velocity,t)
        nonprojector={'rejected':False,'max_clean_error':float((wrong-x).abs().max())}
    except ValueError as e:
        nonprojector={'rejected':True,'reason':str(e)}
    # Oracle noise completion, unlike completion with posterior means.
    rng=np.random.default_rng(21);n=100000
    clean=rng.standard_normal((n,1));noise=rng.standard_normal((n,1));time=.4
    xcoarse=time*clean+(1-time)*noise
    gennoise=rng.standard_normal((n,4))
    lifted=time*clean+(1-time)*(noise/2+gennoise-gennoise.mean(1,keepdims=True))
    innovation=(lifted-time*clean)/(1-time)
    cov_error=float(np.max(np.abs(np.cov(innovation,rowvar=False)-np.eye(4))))
    assert cov_error < .015
    # The zero-conditional-curvature path between the SAME endpoints is the
    # existing endpoint-flow construction, not a new RDD algorithm.
    straight=EndpointFlow(.4,.7,velocity_clock='global')
    x=torch.randn(3,2,8,8,dtype=torch.float64);eps=torch.randn_like(x)
    t=torch.tensor([.4,.55,.7],dtype=x.dtype);u=t[:,None,None,None]
    low=straight.projection(x);high=x-low;out=straight.sample(x,t,eps)
    linear_mean=u*low+.7*(u-.4)/.3*high
    linear_error=float((out.state-(linear_mean+(1-u)*eps)).abs().max())
    assert linear_error<1e-12
    return {'path_audit':rows,'spatiotemporal_projection_audit':temporal_rows,
            'nonprojector_inverse':nonprojector,
            'straight_path_equals_existing_endpoint_flow_max_error':linear_error,
            'oracle_noise_completion_covariance_max_error':cov_error}


class Mixture:
    def __init__(self, stage, gaussian=False):
        self.stage=stage
        self.means=np.array([[.4,-.2]]) if gaussian else np.array([[-1.5,-.8],[.3,1.1],[1.7,-.4]])
        self.variances=np.array([[.8,.2]]) if gaussian else np.array([[.15,.25],[.2,.12],[.25,.18]])
        self.probs=np.array([1.]) if gaussian else np.array([.25,.45,.3])

    def statistics(self,x,t):
        a,da,sigma,_=self.stage.coefficients(t)
        covariance=a*a*self.variances+sigma*sigma
        diff=x[:,None,:]-a*self.means
        logits=np.log(self.probs)-.5*(np.log(covariance)+diff*diff/covariance).sum(-1)
        weight=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        posterior=self.means+self.variances*a/covariance*diff
        clean=(weight[:,:,None]*posterior).sum(1)
        score=-(weight[:,:,None]*diff/covariance).sum(1)
        # E[epsilon|X_t]=-sigma*score; stable even at sigma=0.
        velocity=da*clean+sigma*score
        return clean,score,velocity

    def v(self,x,t):return self.statistics(x,t)[2]

    def sample(self,n,seed):
        rng=np.random.default_rng(seed);labels=rng.choice(len(self.probs),n,p=self.probs)
        clean=self.means[labels]+np.sqrt(self.variances[labels])*rng.standard_normal((n,2))
        a,_,sigma,_=self.stage.coefficients(self.stage.start)
        return a*clean+sigma*rng.standard_normal((n,2))


def integrate(field, initial, method, budget):
    stage=field.stage
    n=budget//2 if method in ('heun','midpoint') else budget
    grid=np.linspace(stage.start,stage.end,n+1)
    x=initial.copy();history=None;pv=None
    for t,s in zip(grid,grid[1:]):
        h=s-t;v=field.v(x,t)
        if method=='euler':x=x+h*v
        elif method=='heun':x=x+h/2*(v+field.v(x+h*v,s))
        elif method=='midpoint':x=x+h*field.v(x+h*v/2,t+h/2)
        elif method=='ab2':x=x+h*(v if pv is None else 1.5*v-.5*pv)
        elif method in ('path1','path2m'):
            x,clean=stage.step(x,v,t,s,history if method=='path2m' else None)
            history=(t,clean)
        else:raise ValueError(method)
        pv=v
    return x


def numerical():
    # Moment formulas checked independently by numerical quadrature.
    max_quad=0.;max_oracle=0.;max_euler=0.
    rng=np.random.default_rng(91)
    for stage in [Stage(0,.4,False),Stage(.4,.7),Stage(.7,1.)]:
        t=stage.start;s=t+.73*(stage.end-t)
        ratio,w0,w1=stage.weights(t,s)
        for k in range(2):
            q0=(1-s)*quad(lambda u:stage.coefficients(u)[3][k]/(1-u)**2,t,s,epsabs=1e-12)[0]
            q1=(1-s)*quad(lambda u:(u-t)*stage.coefficients(u)[3][k]/(1-u)**2,t,s,epsabs=1e-12)[0]
            max_quad=max(max_quad,abs(q0-w0[k]),abs(q1-w1[k]))
        clean=rng.standard_normal((12,2));noise=rng.standard_normal((12,2))
        at,dt,qt,_=stage.coefficients(t);ass,_,qs,_=stage.coefficients(s)
        x=at*clean+qt*noise;v=dt*clean-noise;expected=ass*clean+qs*noise
        actual,_=stage.step(x,v,t,s)
        max_oracle=max(max_oracle,float(np.max(np.abs(actual-expected))))
        max_euler=max(max_euler,float(np.max(np.abs(x+(s-t)*v-expected))))
        if stage.drift:
            formula=x+(s-t)*v+(s-t)**2/(stage.end-stage.start)*np.array([0,1])*clean
            assert np.max(np.abs(formula-actual))<1e-12
    assert max_quad<1e-11 and max_oracle<1e-12
    bench=[];score_error=0.;clean_error=0.;sde_cov_error=0.
    generic_flow_identity_error=0.;naive_true_clean_substitution_error=0.
    for gaussian in [True,False]:
        for stage in [Stage(0,.4,False),Stage(.4,.7),Stage(.7,1.)]:
            field=Mixture(stage,gaussian);x=field.sample(64,20261101)
            reference=solve_ivp(lambda t,y:field.v(y.reshape(x.shape),t).ravel(),
                          (stage.start,stage.end),x.ravel(),method='DOP853',rtol=2e-12,atol=2e-13)
            assert reference.success
            ref=reference.y[:,-1].reshape(x.shape)
            t=(stage.start+stage.end)/2;a,da,sigma,b=stage.coefficients(t)
            m,score,v=field.statistics(x,t)
            recovered=stage.clean(x,v,t)
            effective=x+sigma*v
            generic_flow_identity_error=max(generic_flow_identity_error,
                float(np.max(np.abs((-x+effective)/sigma-v))))
            naive_true_clean_substitution_error=max(naive_true_clean_substitution_error,
                float(np.max(np.abs((-x+recovered)/sigma-v))))
            clean_error=max(clean_error,float(np.max(np.abs(recovered-m))))
            score_error=max(score_error,float(np.max(np.abs(-(x-a*recovered)/sigma**2-score))))
            if gaussian:
                var=field.variances[0];cov=a*a*var+sigma**2
                vc=a*da*var-sigma;g2=.27
                # Covariance evolution of dX=(v+.5*g^2*score)dt+g dW.
                sde_derivative=2*(vc/cov-.5*g2/cov)*cov+g2
                sde_cov_error=max(sde_cov_error,float(np.max(np.abs(sde_derivative-2*vc))))
            errors={}
            for method in ['euler','heun','midpoint','ab2','path1','path2m']:
                entries=[]
                for budget in [4,8,16,32,64,128]:
                    value=integrate(field,x,method,budget)
                    entries.append({'nfe':budget,'rmse':float(np.sqrt(np.mean((value-ref)**2)))})
                slope=math.log(entries[-2]['rmse']/entries[-1]['rmse'],2)
                errors[method]={'errors':entries,'observed_order_64_to_128':slope}
            bench.append({'field':'gaussian' if gaussian else 'three_component_mixture',
                          'stage':[stage.start,stage.end],'drift':stage.drift,'initials':len(x),
                          'reference_nfe':reference.nfev,'methods':errors})
    assert score_error<1e-10 and clean_error<1e-10 and sde_cov_error<1e-12
    # Interior convergence: quadratic path does not make path1 second order.
    interior=bench[1]['methods']
    assert .8 < interior['path1']['observed_order_64_to_128'] < 1.2
    assert 1.8 < interior['path2m']['observed_order_64_to_128'] < 2.2
    return {'quadrature_max_error':max_quad,'constant_clean_one_step_max_error':max_oracle,
            'constant_clean_euler_max_error':max_euler,'bayes_clean_inverse_max_error':clean_error,
            'bayes_score_max_error':score_error,'sde_covariance_identity_max_error':sde_cov_error,
            'generic_flow_identity_max_error':generic_flow_identity_error,
            'naive_true_clean_substitution_max_error':naive_true_clean_substitution_error,
            'benchmarks':bench}


def boundary():
    # Exact Gaussian Bayes denoiser; errors below persist with a perfect net.
    a=.4;sigma=1-a;tau2=1.;m=4
    var_z=a*a*tau2+sigma*sigma
    var_mean=a*a*tau2*tau2/var_z
    posterior_variance=tau2-var_mean
    repeat=np.ones((m,1));p=repeat@repeat.T/m;q=np.eye(m)-p
    desired=a*a*tau2*(repeat@repeat.T)+sigma*sigma*np.eye(m)
    fresh=a*a*var_mean*(repeat@repeat.T)+sigma*sigma*np.eye(m)
    deficit=desired-fresh
    expected=a*a*posterior_variance*(repeat@repeat.T)
    assert np.max(np.abs(deficit-expected))<1e-12
    # Actual linear mapping of the same coarse z with posterior mean and
    # posterior innovation; these are correlated and innovation is not IID.
    cm=a*tau2/var_z;ce=sigma/var_z
    noise_fill=var_z*(a*cm+sigma*ce/math.sqrt(m))**2*(repeat@repeat.T)+sigma*sigma*q
    # A separate normalized-coarse coordinate design has an exact lift kernel.
    normalized_cov=(a*a*m*tau2+sigma*sigma)*p+sigma*sigma*q
    assert np.max(np.abs(normalized_cov-desired))<1e-12
    return {'boundary':a,'coarse_clean_variance':tau2,'bayes_clean_variance':var_mean,
            'posterior_variance':posterior_variance,
            'target_fine_diagonal':np.diag(desired).tolist(),
            'fresh_fine_diagonal':np.diag(fresh).tolist(),
            'fresh_relative_trace_deficit':float(np.trace(deficit)/np.trace(desired)),
            'isometric_with_bayes_innovation_diagonal':np.diag(noise_fill).tolist(),
            'normalized_coarse_design_covariance_error':float(np.max(np.abs(normalized_cov-desired))),
            'normalized_design_requires_different_coarse_training_normalization':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    report={'scope':'CPU float64 analytic fields; no trained network quality measurements',
            'algebra':algebra(),'numerical':numerical(),'boundary_counterexample':boundary(),
            'torch':torch.__version__,'numpy':np.__version__,'cuda_initialized':torch.cuda.is_initialized()}
    paths=['unified_multires/research/rdd_fm_integrator.py','unified_multires/tools/validate_rdd_fm_solver_20261101.py',
           'unified_multires/umdiff/paths.py','multiscale_diffusion/diffusion/flow_matching_with_drift.py',
           'multiscale_diffusion/rdd_video/flow_matching_video.py','FastVideo-RDD-minimal/rdd/flow.py']
    report['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(out),'cuda_initialized':report['cuda_initialized'],
          'nonprojector':report['algebra']['nonprojector_inverse'],
          'oracle_error':report['numerical']['constant_clean_one_step_max_error'],
          'boundary_trace_deficit':report['boundary_counterexample']['fresh_relative_trace_deficit']}))


if __name__=='__main__':main()
