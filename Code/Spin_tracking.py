import numpy as np
import linecache
from typing import Optional

def compute_F_cost_matrix(
    F_curr: np.ndarray,         # (nst, ndof) 可能为复数；代码仅用其实部
    F_prev: np.ndarray,         # (nst, ndof) 可能为复数；这里用于线性外推
    e_curr: np.ndarray,         # (nst,) 本步各态能量（对角元）
    e_prev: np.ndarray,         # (nst,) 上一步各态能量（对角元）
    criteria,                        #  cost矩阵
    count,                      # 记录积分步数
    dt: float,                  # 时间步长
    act_state: Optional[int] = None,  # 当前活跃态；仅在做半步修正时需要
    *,
    sign_eps: float = 1e-10,    # 符号判断的零阈值：|x|<eps 视为 0
    correct_half_step: bool = False   # 是否做“半步动量已被旧力推进”的修正
) -> np.ndarray:
    """
    计算 F-tracking 的“代价/权重矩阵”（行和为 1 的记分板）：
    对每一对态 (i,j)，用上一时刻的能隙 ΔE_ij(t) 与利用力的线性外推
    得到的 ΔE_ij(t+dt) 的“符号是否翻转”来度量 (i→j) 的配对偏好。

    返回:
        criteria: (nst, nst) 的实数矩阵；每一行和为 1。非对角元素越大，越支持
             将上一时刻的态 i 对应到当前步的态 j（发生顺序翻转/近简并）。
    """

    F_curr = np.asarray(F_curr)
    F_prev = np.asarray(F_prev)
    e_prev = np.asarray(e_prev)
    criteria = np.asarray(criteria)
    count = np.asarray(count)
    #print(res)
    #print(F_curr)
    # --------------------------
    # 自定义带阈值的符号函数：避免 |x| 很小的数值噪声
    # 返回值 ∈ {-1, 0, +1}
    # --------------------------
    def sgn(x: float, eps: float = sign_eps) -> int:
        if x > eps:  return 1
        if x < -eps: return -1
        return 0

    # --------------------------
    # 主循环：为每个 (i,j) 计算 “翻转指示” val = 1 - sign(ΔE) * sign(ΔE_new)
    # 并将每行的对角元设为 1 - sum(非对角)
    # --------------------------
    res = np.zeros((state_num, state_num), dtype=float)

    # 只用上一时刻的力做线性外推（与原 C++ 完全一致）
    Fp_real = np.real(F_prev)  

    for i in range(state_num):
        row_sum = 1.0
        Ei = e_prev[i]
        Fi = Fp_real[i]

        # 逐 j 计算 val，并累计非对角到 row_sum
        for j in range(state_num):
            if i == j:
                val = 0.0  # 按原实现：对角在循环结束后再统一设置
            else:
                Ej = e_prev[j]
                Fj = Fp_real[j]

                # 上一步能隙 ΔE_ij(t)
                dE = Ej - Ei
                
                if abs(dE) > 0.0001 and abs(dE) <0.05 and abs(Fj) < 0.01 and abs(Fi) < 0.01:              
                    # 一阶外推：E_k(t+dt) ≈ E_k(t) - F_k · Δq
                    dE_i = -np.dot(Fi, dt)   # i 态能量改变量
                    dE_j = -np.dot(Fj, dt)   # j 态能量改变量
                    dE_new = dE + dE_j - dE_i
                    factor = dE/(dE_j-dE_i)

                    
                    if factor > 0 :
                        if criteria[i,j] >= 1:
                            res[i,j] = 1
                            criteria[i,j] = 0
                            count[i,j] = 0
                        else:
                            criteria[i,j] += (dE_j-dE_i)/dE
                            count[i,j] += 1
                        
                        
                    else:
                        criteria[i,j] = 0
                        count[i,j] = 0
                        
                        
                    #else:
                    #    if criteria[i,j] >= 1:
                    #        res[i,j] = 1
                    #        criteria[i,j] = 0
                    #        count[i,j] = 0
                    #    else:
                    #        criteria[i,j] = 0
                    #        count[i,j] = 0
                    #print(res)
                    
    

    return res,criteria,count
    
    
if __name__ == "__main__":
    N_kpt = 21
    time_step = 970 #时间步数
    state_num = 6    #每个k点上的电子态个数
    spin_state = 2   #spin-tracking的电子态个数
    dt = 1  # 仅示意
    inital_num = 9001 #nac文件起始数
    act_state = state_num
    order = np.tile(np.arange(state_num*N_kpt), (time_step, 1))
    a = open('Energy_reordered_spin_tracking.dat','w+')
    b = open('Mag_reordered_spin_tracking.dat','w+')
    
    
    Energy_total = np.zeros((time_step,state_num*N_kpt))
    Mag_total = np.zeros((time_step,state_num*N_kpt))
    for nkpt in range(N_kpt):
    
        # 读取Energy并初始化
        Energy_origin = np.zeros((time_step,state_num))
        Energy_reorder = np.zeros((time_step,state_num))
        Mag_origin = np.zeros((time_step,state_num))
        Mag_reorder = np.zeros((time_step,state_num))
        for i in range(time_step):
            for j in range(state_num):
                a1=linecache.getlines('energy')[i]
                E_j = a1.split()[(nkpt)*6+j]
                Energy_origin[i,j]=float(E_j)
                a2=linecache.getlines('Mag_ext_origin.dat')[i]
                Mag = a2.split()[(nkpt)*6+j]
                Mag_origin[i,j]=float(Mag)
                
        for i in range(time_step):
            if Mag_origin[i,2] < 0:
                #Mag 矫正
                temp = Mag_origin[i,2].copy()
                Mag_origin[i,2] = Mag_origin[i,3] 
                Mag_origin[i,3] = temp
                
                #Energy 矫正
                temp = Energy_origin[i,2].copy()
                Energy_origin[i,2] = Energy_origin[i,3]
                Energy_origin[i,3] = temp
                
                #order 矫正
                temp = order[nkpt*state_num+2].copy()
                order[nkpt*state_num+2] = order[nkpt*state_num+3]
                order[nkpt*state_num+3] = temp
                
        if nkpt > 10:
            Energy_origin = Energy_origin+0.005
        Energy_total[:,(nkpt)*state_num:(nkpt+1)*state_num] = Energy_origin[:,:]
        Mag_total[:,(nkpt)*state_num:(nkpt+1)*state_num] = Mag_origin[:,:]
    
                        
    #NAC矩阵重排
    
    #for i in range(time_step):
    #    nac_file = open('Spin_real%04d' % (inital_num+i),'w+')
    #    nac_origin = np.zeros((state_num*N_kpt, 2*state_num*N_kpt))
    #    for j in range(state_num*N_kpt):
    #        for k in range(state_num*N_kpt):
    #            a2=linecache.getlines('real_%d' % (i+inital_num))[j]
    #            nac_origin[j,2*k] = float(a2.split()[2*k])
    #            nac_origin[j,2*k+1] = float(a2.split()[2*k+1])
    #            
    #    nac_reordered = np.zeros((state_num*N_kpt, 2*state_num*N_kpt))
    #    nac_temp = np.zeros((state_num*N_kpt, 2*state_num*N_kpt))
    #    for j in range(state_num*N_kpt):
    #        nac_temp[j,:]=nac_origin[order[i,j],:].copy()
    #        
    #    for j in range(state_num*N_kpt):
    #        nac_reordered[:,2*j] = nac_temp[:,2*order[i,j]].copy()
    #        nac_reordered[:,2*j+1] = nac_temp[:,2*order[i,j]+1].copy()
    #        
    #    for j in range(state_num*N_kpt):
    #        for k in range(state_num*N_kpt):
    #            print(nac_reordered[j,2*k],'     ',nac_reordered[j,2*k+1],end='     ',file=nac_file)
    #        print(' ',file=nac_file)
    #    print('real trans done %d'%i)
            
    
    for i in range(time_step):
        for j in range(N_kpt*state_num):
            print(Energy_total[i,j],end='    ',file = a)                
            print(Mag_total[i,j],end='    ',file = b)
        print('  ',file = a)
        print('  ',file = b)
   
    
    

