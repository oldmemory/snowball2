# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:20:24 2021

@author: these
"""



# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:12:05 2021

@author: these
"""

basedir=r'D:\西南证券\衍生品\定价\雪球\程序'
import numpy as np
import pandas as pd
import os
os.chdir(basedir)
from pre_param import *
Supout=103
S0=100
r=0.05
b=0.05
sigma=0.2

#intra_day_noofseqs仅用于估值当天。
intra_day_noofseqs=4

#5%的波动率，一个小时的相对波动方差就在正负np.exp(0.05/np.sqrt(244)/np.sqrt(4))-1大约0.15%.
#我们可以再细一点，设为0.015%，为了能够整除，所以我们选
down_interval_ratios=[1e-4,1e-4,2e-4]
#当然如果考虑drift，(0.02-1/2*sigma**2)/252/4=1.5e-5
#事实上我们可以把这么细的仅用于

#在计算宽度方面，我们以50%的波动率来计算，因为正常情况下一般很少到50%。对雪球，我们r=0来计算股价下边界
#np.exp(0*2-0.5*np.sqrt(2)*3)
#2年期限，St/S0,3倍方差为12%(我们取10%)，4倍方差为6%，5倍方差为3%
down_ranges=[1,0.1,0.06,0.03]


def gen_S_intervals(up_ranges,up_interval_ratios,down_ranges,down_interval_ratios):
    up_result=[]
    down_result=[]
    result=[]
    for i in range(1,len(up_ranges)):
        temp_num=int(np.abs(up_ranges[i]-up_ranges[i-1])/up_interval_ratios[i-1])
        temp=np.linspace(up_ranges[i-1],up_ranges[i], num=temp_num,endpoint=False)
        up_result=np.concatenate([up_result,temp])
    result=np.concatenate([result,up_result])
    
    for i in range(1,len(down_ranges)):
        temp_num=int(np.abs(down_ranges[i]-down_ranges[i-1])/down_interval_ratios[i-1])
        temp=np.linspace(down_ranges[i-1],down_ranges[i], num=temp_num,endpoint=False)
        down_result=np.concatenate([down_result,temp])
    temp=np.flipud(down_result[1:])
    result=np.concatenate([temp,result])
    
    return(result)


#忽略尾部小于等于最小值的部分，忽略高于或者等于H的部分
#在计算上边界的时候，因为上边界只需要计算到upout，所以可以直接用
up_ranges=[1,1.04]
up_interval_ratios=[1e-4]
Ss=gen_S_intervals(up_ranges,up_interval_ratios,down_ranges,down_interval_ratios)
H_index=np.where(Ss==Supout/S0)[0][0]

#两端对称：忽略尾部小于等于最小值或者大于等于最大值的部分
#np.exp(0*2+0.5*np.sqrt(2)*5)
up_ranges=[1,10,17,34]
up_interval_ratios=down_interval_ratios
Ss=gen_S_intervals(up_ranges,up_interval_ratios,down_ranges,down_interval_ratios)
H_index=np.where(Ss==Supout/S0)[0][0]
#knock_out_dates_reverse=[30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,\
#                 540,570,600,630,660]

pricing_seqs=[]
#日期第一个可以是小数，后面必须都是整数

temp=pd.read_excel('敲出日期.xlsx',header=None)
KO_trd_dates=temp[0].tolist()
noofsep_dates=len(KO_trd_dates)-1
sep_trd_dates=[tday_noofdays(KO_trd_dates[i-1],KO_trd_dates[i]) for i in range(1,len(KO_trd_dates))]
KO_natural_dates=[(KO_trd_dates[i]-KO_trd_dates[0]).days for i in range(1,len(KO_trd_dates))]
sep_natural_dates=[(KO_trd_dates[i]-KO_trd_dates[i-1]).days for i in range(1,len(KO_trd_dates))]

#日期第一个可以是小数，后面必须都是整数
noofsep_dates=22
KO_trd_dates=np.linspace(start=48,stop=244*2,num=noofsep_dates,dtype=int)
sep_trd_dates=[KO_trd_dates[0]]+[KO_trd_dates[i]-KO_trd_dates[i-1] for i in range(1,len(KO_trd_dates))]
KO_natural_dates=np.linspace(start=100,stop=730,num=noofsep_dates,dtype=int)
sep_natural_dates=[KO_natural_dates[0]]+[KO_natural_dates[i]-KO_natural_dates[i-1] for i in range(1,len(KO_natural_dates))]

#测试：日期第一个可以是小数，后面必须都是整数
noofsep_dates=3
KO_trd_dates=np.linspace(start=10,stop=30,num=noofsep_dates,dtype=int)
sep_trd_dates=[KO_trd_dates[0]]+[KO_trd_dates[i]-KO_trd_dates[i-1] for i in range(1,len(KO_trd_dates))]
KO_natural_dates=np.linspace(start=15,stop=45,num=noofsep_dates,dtype=int)
sep_natural_dates=[KO_natural_dates[0]]+[KO_natural_dates[i]-KO_natural_dates[i-1] for i in range(1,len(KO_natural_dates))]

#在BS框架下
from scipy.stats import norm
cdf1=norm.cdf
pdf1=norm.pdf
#part I：只有敲出
KO_payoff=[1 for i in range(noofsep_dates)]#green函数
KO_payoff=np.multiply(0.01/naturalday_year,KO_natural_dates).tolist()#年化1%
no_KO_payoff=0

pricing_seqs=[[no_KO_payoff for Sj in Ss[:H_index] ]]##期末。
for i_day in range(noofsep_dates):
    delta_T_trd=sep_trd_dates[-(i_day+1)]/tday_year
    delta_T_natural=sep_natural_dates[-(i_day+1)]/naturalday_year
    DF=np.exp(-r*delta_T_trd)
    temp_result=[]
    if i_day<noofsep_dates-1:#非定价日
        max_index=H_index#不包括H
    else:
        max_index=len(Ss)#包括所有
    for Sj_index in range(max_index):
        Sj=Ss[Sj_index]
        H_std=(np.log(Supout/S0/Sj)-(b-sigma**2/2)*delta_T_trd)/sigma/np.sqrt(delta_T_trd)
        temp=(1-cdf1(H_std))*KO_payoff[-(i_day+1)]
        
        Sjp1_std=(np.log(Ss[:(H_index+1)]/Sj)-(b-sigma**2/2)*delta_T_trd)/sigma/np.sqrt(delta_T_trd)
        temp=temp+np.sum([( cdf1(Sjp1_std[j])-cdf1(Sjp1_std[j-1]) )*pricing_seqs[-1][j-1] for j in range(1,len(Sjp1_std))])
        
        temp=temp*DF
        temp_result.append(temp)
    pricing_seqs.append(temp_result)
    print(i_day)

#part I：无敲出票息，但有红利票息
KO_payoff=[0 for i in range(noofsep_dates)]#green函数
no_KO_payoff=0.01*KO_natural_dates[-1]/naturalday_year#年化1%

pricing_seqs=[[no_KO_payoff for Sj in Ss[:H_index] ]]##期末。
for i_day in range(noofsep_dates):
    delta_T_trd=sep_trd_dates[-(i_day+1)]/tday_year
    delta_T_natural=sep_natural_dates[-(i_day+1)]/naturalday_year
    DF=np.exp(-r*delta_T_trd)
    temp_result=[]
    if i_day<noofsep_dates-1:#非定价日
        max_index=H_index#不包括H
    else:
        max_index=len(Ss)#包括所有
    for Sj_index in range(max_index):#不包括H
        Sj=Ss[Sj_index]
        H_std=(np.log(Supout/S0/Sj)-(b-sigma**2/2)*delta_T_trd)/sigma/np.sqrt(delta_T_trd)
        temp=(1-cdf1(H_std))*KO_payoff[-(i_day+1)]
        
        Sjp1_std=(np.log(Ss[:(H_index+1)]/Sj)-(b-sigma**2/2)*delta_T_trd)/sigma/np.sqrt(delta_T_trd)
        temp=temp+np.sum([( cdf1(Sjp1_std[j])-cdf1(Sjp1_std[j-1]) )*pricing_seqs[-1][j-1] for j in range(1,len(Sjp1_std))])
        
        temp=temp*DF
        temp_result.append(temp)
    pricing_seqs.append(temp_result)
    print(i_day)
    
##payoff函数
def payoff_KO(S_N,H=0):
    if S_N>H:
        return(1)
    else:
        return(0)
def payoff_noKOnpKI_1(S_N,H=0,L=0):
    if S_N>H:
        return(0)
    elif S_N<L:
        return(0)
    else:
        return(1)
def payoff_noKOnpKI_2(S_N,H=0,L=0,payoff_temp,K):
    if S_N>H:
        return(0)
    elif S_N<L:
        return(0)
    else:
        return(payoff_temp(S_N,K))
        



