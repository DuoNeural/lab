import numpy as np, matplotlib, json, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

THETA = np.array([-3.32464574, 2.99365192, 1.87992746, 1.82114601])

def Rx(p): return np.array([[np.cos(p/2),-1j*np.sin(p/2)],[-1j*np.sin(p/2),np.cos(p/2)]],dtype=complex)
def Ry(p): return np.array([[np.cos(p/2),-np.sin(p/2)],[np.sin(p/2),np.cos(p/2)]],dtype=complex)
def Rz(p): return np.array([[np.exp(-1j*p/2),0],[0,np.exp(1j*p/2)]],dtype=complex)

def reset0(rho):
    r=np.zeros_like(rho)
    r[0,0]=rho[0,0]+rho[1,1]; r[0,2]=rho[0,2]+rho[1,3]
    r[2,0]=rho[2,0]+rho[3,1]; r[2,2]=rho[2,2]+rho[3,3]
    return r

CX=np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],dtype=complex)
U_ans=(np.kron(Rz(THETA[3]),np.eye(2))@np.kron(np.eye(2),Rz(THETA[2])))@CX@(np.kron(Ry(THETA[1]),np.eye(2))@np.kron(np.eye(2),Ry(THETA[0])))
RX={0:np.kron(np.eye(2,dtype=complex),Rx(0.0)),1:np.kron(np.eye(2,dtype=complex),Rx(np.pi))}

def qrnn(seq):
    rho=np.zeros((4,4),dtype=complex); rho[0,0]=1.0
    for x in seq:
        rho=reset0(rho); rho=RX[x]@rho@RX[x].conj().T; rho=U_ans@rho@U_ans.conj().T
    return (rho[2,2]+rho[3,3]).real

def gen_seqs(T,N=512):
    if T<=8:
        s=[[((i>>j)&1) for j in reversed(range(T))] for i in range(2**T)]
        t=[float(sum(x)%2) for x in s]
        return np.array(s),np.array(t)
    rng=np.random.RandomState(42+T); s,t,c=[],[],{0:0,1:0}
    while len(s)<N:
        x=list(rng.randint(0,2,T)); tgt=int(sum(x)%2)
        if c[tgt]<N//2: s.append(x); t.append(float(tgt)); c[tgt]+=1
    return np.array(s),np.array(t)

def edecay(t,M,tau): return M*np.exp(-t/tau)

print("="*60); print("Q-DHP VALIDATION  T=3..100"); print("="*60)
print(f"{'T':>4}|{'Acc%':>8}|{'Margin':>9}")
Ls=list(range(3,101)); accs=[]; margs=[]
for T in Ls:
    X,y=gen_seqs(T)
    p=np.array([qrnn(list(x.astype(int))) for x in X])
    acc=float(np.mean((p>0.5).astype(int)==y.astype(int))*100)
    mg=float(np.mean(np.abs(p-0.5))*2)
    accs.append(acc); margs.append(mg)
    flag=""
    if T>3 and accs[-2]>=99.9 and acc<99.9: flag=" <-- 100% cliff"
    elif T>3 and accs[-2]>=55.0 and acc<55.0: flag=" <-- ~chance"
    print(f"{T:>4}|{acc:>7.2f}%|{mg:>9.6f}{flag}")

La=np.array(Ls,dtype=float); ma=np.array(margs); aa=np.array(accs)
ts100=next((Ls[i]-1 for i,a in enumerate(accs) if a<99.9),None)
ts55=next((Ls[i] for i,a in enumerate(accs) if a<55.0),None)
valid=ma>0.01; tL_exp=None; M0=None
try:
    po,_=curve_fit(edecay,La[valid],ma[valid],p0=[1.0,30.0],maxfev=5000)
    M0,tL_exp=float(po[0]),float(po[1])
except Exception as e: print(f"fit err:{e}")
tL_ll=None
try:
    sl,ic,_,_,_=linregress(La[valid],np.log(ma[valid]+1e-12))
    if sl<0: tL_ll=float(-1/sl)
except: pass

print("\n"+"="*60+"  RESULTS")
print(f"  tau*(100%) = {ts100}")
print(f"  tau*(55%)  = {ts55}")
if tL_exp: print(f"  tau_L expfit = {tL_exp:.2f}")
if tL_ll:  print(f"  tau_L loglin = {tL_ll:.2f}")
for tsl,tsv in [("100%",ts100),("55%",ts55)]:
    for tll,tlv in [("exp",tL_exp),("ll",tL_ll)]:
        if tsv and tlv:
            r=tsv/tlv; d=abs(r-0.72)
            match=" <<< MATCH 0.72" if d<0.05 else ""
            print(f"  tau*({tsl})/tau_L({tll})={tsv}/{tlv:.1f}={r:.4f} delta={d:.4f}{match}")

os.makedirs("/home/ai/duoneural/aura",exist_ok=True)
with open("/home/ai/duoneural/aura/q_dhp_validate_results.json","w") as f:
    json.dump({"lengths":Ls,"gen_accs":accs,"gen_margins":margs,"tau_star_100":ts100,"tau_star_55":ts55,"tau_L_expfit":tL_exp,"tau_L_loglin":tL_ll},f,indent=2)

plt.style.use('dark_background')
fig,(a1,a2)=plt.subplots(2,1,figsize=(14,9))
a1.plot(Ls,accs,color='#00d2ff',lw=2,label='Gen Accuracy')
if ts100: a1.axvline(ts100,color='#ffbb00',ls='--',lw=1.5,label=f'tau*(100%)={ts100}')
if ts55:  a1.axvline(ts55,color='#ff007f',ls='--',lw=1.5,label=f'tau*(55%)={ts55}')
a1.axhline(50,color='white',lw=0.6,ls=':',alpha=0.3); a1.legend(); a1.grid(alpha=0.2)
a1.set_title('Q-DHP Generalization Sweep — Fixed T=3 Weights',fontsize=13); a1.set_xlim(3,100)
a2.semilogy(Ls,ma+1e-6,color='#ffbb00',lw=2,label='Margin')
if tL_exp:
    tf=np.linspace(3,100,300); a2.semilogy(tf,edecay(tf,M0,tL_exp),color='#a020f0',ls='--',lw=2,label=f'tau_L={tL_exp:.1f}')
if ts55: a2.axvline(ts55,color='#ff007f',ls='--',lw=1.5)
if ts55 and tL_exp:
    r=ts55/tL_exp
    a2.text(0.6,0.15,f'tau*/tau_L={ts55}/{tL_exp:.1f}={r:.3f}',transform=a2.transAxes,fontsize=12,color='#00ff88',bbox=dict(boxstyle='round',fc='#111',alpha=0.7))
a2.legend(); a2.grid(alpha=0.2); a2.set_xlim(3,100)
a2.set_xlabel('Sequence Length T',fontsize=12)
plt.tight_layout()
plt.savefig('/home/ai/duoneural/quantum/q_dhp_validate.png',dpi=150)
print("\nplot -> /home/ai/duoneural/quantum/q_dhp_validate.png")
