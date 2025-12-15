function [bestPath, Fmin, pathRecord, position] = improvedBatPlanning( ...
        threat, r, startX, startY, endX, endY, gridCount, params)
% -------------------------------------------------------------------------
% 改进蝙蝠算法：用于无人机路径规划
% 输入：
%   threat      — n×3  威胁中心 (x,y,0)
%   r           — n×1  威胁半径
%   startX/Y    — 起点坐标
%   endX/Y      — 终点坐标
%   gridCount   — 路径离散段数
%   params      — 结构体，见下方字段
% 输出：
%   bestPath    — 1×(2*gridCount) 最优路径向量
%   Fmin        — 1×N_gen         每代最优适应度曲线
%   pathRecord  — 1×(gridCount+1) 最优路径的回溯索引
%   position    — (gridCount+1)×2 最优路径坐标
% -------------------------------------------------------------------------

% ============== 读取参数 ==============
pop     = params.pop;      N_gen  = params.N_gen;
Qmin    = params.Qmin;     Qmax   = params.Qmax;
Rmax    = params.Rmax;     c1     = params.c1;
c2      = params.c2;       w      = params.w;
F0      = params.F0;       CR     = params.CR;
pathMin = params.pathMin;  pathMax = params.pathMax;
aAtt    = 0.9;             % 响度衰减系数

% 变量初始化
V   = zeros(pop, 2*gridCount);      % 速度
Q   = zeros(pop, 1);               % 频率
A   = rand(pop, 2*gridCount);      % 响度
Rr  = rand(pop, 2*gridCount);      % 脉冲率
Fmin = zeros(1, N_gen);            % 收敛曲线

% --- 初始化种群路径 ---
path = zeros(pop, 2*gridCount);
for i = 1:pop
    % x 坐标均匀分布，y 坐标随机
    for j = 1:gridCount
        path(i, 2*j-1) = startX + j*(endX-startX)/(gridCount+1);
        path(i, 2*j  ) = startY + rand()*(endY-startY);
    end
end

% 计算初始适应度
fitness = zeros(pop,1);
for i = 1:pop
    fitness(i) = verify(path(i,:), threat, r, ...
                        startX, startY, endX, endY, gridCount);
end
[bestFitness, bestIdx] = min(fitness);
bestPath = path(bestIdx,:);

% ------------  迭代  ------------
for t = 1:N_gen
    for i = 1:pop
        % ① 更新频率、速度、位置
        Q(i)   = Qmin + (Qmax-Qmin)*rand();
        V(i,:) = V(i,:) + (bestPath - path(i,:))*Q(i);
        S      = path(i,:) + V(i,:);
        S(S > pathMax) = pathMax;
        S(S < pathMin) = pathMin;

        % ② 随机扰动（局部搜索）
        if rand > mean(Rr(i,:))
            S = bestPath + mean(A(i,:)) * 6^(-30*(1-t/N_gen)^4);
        end

        % ③ 早熟判断 + DE/PSO 混合
        if std(fitness) < 1e-3          % 可根据需要调整阈值
            dx = randperm(pop,3);
            F  = F0 * 2.^exp((1-N_gen)/(N_gen+1-t));
            V(i,:) = w*S + c1*rand*(bestPath-S) + ...
                     c2*F*(path(dx(2),:)-path(dx(3),:));
            if rand > CR, S = V(i,:); end
        end

        % ④ 适应度评估
        newFit = verify(S, threat, r, ...
                        startX, startY, endX, endY, gridCount);

        % ⑤ 接受准则
        if (newFit <= fitness(i)) && (rand < mean(A(i,:)))
            path(i,:)   = S;
            fitness(i)  = newFit;
            A(i,:)      = aAtt * A(i,:);
            Rr(i,:)     = Rmax * (1 - exp(-0.9*t));
        end

        % ⑥ 更新全局最优
        if newFit < bestFitness
            bestFitness = newFit;
            bestPath    = S;
        end
    end
    Fmin(t) = bestFitness;
end

% ---------- 输出最优路径坐标 ----------
[~, pathRecord, position] = verify(bestPath, threat, r, ...
                                  startX, startY, endX, endY, gridCount);
end
