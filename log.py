import os

# 1. 基础 MMS 验证实验(Case 1: 圆形界面)
# os.system('python run.py --case mms --config configs/case1/beta-m-p-1-10.yaml')
# os.system('python run.py --case mms --config configs/case1/beta-m-p-1-100.yaml')
# os.system('python run.py --case mms --config configs/case1/beta-m-p-10-1.yaml')

# 2. 复杂物理场测试 (Case 2: 角向振荡解)
# os.system('python run.py --case oscillating --config configs/case2/config.yaml')

# 3. 复杂几何界面测试 (Case 3: 椭圆界面)
# os.system('python run.py --case elliptic --config configs/case3/config.yaml')


# 4. 收敛性研究 
os.system('python run.py --case convergence --config configs/convergence/config.yaml')
