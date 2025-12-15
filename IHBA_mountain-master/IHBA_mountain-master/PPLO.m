function [Best_score, Best_pos, Convergence_curve] = PPLO(pop, T, lb, ub, dim, fobj)
    % 超参数设置
    groups = 6; 
    Np = floor(pop / groups);
    Lb = lb;  % ✅ 修正为 1×dim 向量
    Ub = ub;
    elite_pool_size = 5;  % 精英池容量

    
    % 改进1：Sobol序列初始化（比 LHS 具有更好的空间填充性）
    initial_pop = net(sobolset(dim), pop);
    initial_pop = initial_pop .* (Ub - Lb) + Lb;
    
    % 初始化子群
    for g = 1:groups
        idx = (g-1)*Np + 1 : g*Np;
        group(g).pos = initial_pop(idx, :);
        group(g).vel = zeros(Np, dim);
        group(g).fitness = inf(Np, 1);
        group(g).pBest = group(g).pos;
        group(g).pBestScore = inf(Np, 1);
        group(g).gBestScore = inf;
        group(g).gBest = zeros(1, dim);
    end
    
    % 改进2：精英档案库初始化
    elite_pool = repmat(struct('position',[],'fitness',inf), 1, elite_pool_size);
    stagnation_counter = 0;  % 停滞检测计数器
    
    % 全局最优解初始化
    Destination_fitness = inf;
    Destination_position = zeros(1, dim);
    Convergence_curve = zeros(1, T);
    last_fitness = inf;
    
    % 主循环
    for t = 1:T
        % 改进3：非线性参数衰减
        E = 1.0 / (1 + exp(0.1*(t - T/2)));  % Sigmoid 衰减曲线
        Inertia = 0.9 * (1 - tanh(5*t/T));    % 非线性惯性权重
        search_radius = 0.1*(Ub - Lb)*exp(-4*t/T);  % 邻域收缩
        
        % 停滞检测（超过10代无改进触发强扰动）
        if abs(last_fitness - Destination_fitness) < 1e-8
            stagnation_counter = stagnation_counter + 1;
            if stagnation_counter > 10
                % 触发强扰动：在精英解周围进行高斯扰动
                for k = 1:elite_pool_size
                    if ~isempty(elite_pool(k).position)
                        perturb_pos = elite_pool(k).position + 0.05*randn*(Ub-Lb);
                        perturb_pos = max(min(perturb_pos, Ub), Lb);
                        perturb_fit = fobj(perturb_pos);
                        if perturb_fit < Destination_fitness
                            Destination_position = perturb_pos;
                            Destination_fitness = perturb_fit;
                        end
                    end
                end
                stagnation_counter = 0;
            end
        else
            stagnation_counter = 0;
        end
        
        % 子群间交互（动态频率）
        interact_freq = 5 + floor(15/(1 + exp(-0.1*(t - T/3))));
        if mod(t, interact_freq) == 0
            % 多模态精英交叉
            for g = 1:groups
                % 从精英池中选择交叉对象
                partner = elite_pool(randi(elite_pool_size));
                if ~isempty(partner.position)
                    alpha = 0.4 + 0.2*rand;
                    cross_pos = alpha*group(g).gBest + (1-alpha)*partner.position;
                    cross_pos = cross_pos + 0.05*randn*(Ub-Lb);
                    cross_fit = fobj(cross_pos);
                    if cross_fit < Destination_fitness
                        Destination_fitness = cross_fit;
                        Destination_position = cross_pos;
                    end
                end
            end
        end
        
        % 遍历每个子群
        for g = 1:groups
            % 改进4：拟牛顿方向修正（后期局部搜索）
            if t > 2*T/3
                % 计算数值梯度（逐维有限差分）
                base_val = fobj(group(g).gBest);
                eps_val = 1e-6;
                grad = zeros(1, dim);
                for j = 1:dim
                    temp = group(g).gBest;
                    temp(j) = temp(j) + eps_val;
                    grad(j) = (fobj(temp) - base_val) / eps_val;
                end
                newton_dir = -grad;
                newton_dir = newton_dir / norm(newton_dir);
            end
            
            for i = 1:Np
                % 阶段自适应速度更新
                if t < T/3
                    % 全局探索阶段
                    levy_step = levy(1, dim, 1.8);
                    group(g).vel(i,:) = Inertia*group(g).vel(i,:) + ...
                        2.0*rand*(Destination_position - group(g).pos(i,:)) + ...
                        0.6*levy_step;
                elseif t < 2*T/3
                    % 局部开发阶段
                    brownian = 0.1*randn(1,dim).*(Ub-Lb);
                    group(g).vel(i,:) = Inertia*group(g).vel(i,:) + ...
                        1.2*rand*(group(g).pBest(i,:) - group(g).pos(i,:)) + ...
                        0.8*rand*(group(g).gBest - group(g).pos(i,:)) + ...
                        brownian;
                else
                    % 精细搜索阶段
                    if exist('newton_dir','var')
                        group(g).vel(i,:) = 0.2*group(g).vel(i,:) + ...
                            0.6*newton_dir.*(Ub-Lb) + ...
                            0.1*randn*(Ub-Lb);
                    else
                        group(g).vel(i,:) = 0.2*group(g).vel(i,:) + ...
                            0.8*rand*(Destination_position - group(g).pos(i,:));
                    end
                end
                
                % 位置更新与邻域收缩
                new_pos = group(g).pos(i,:) + group(g).vel(i,:);
                new_pos = max(min(new_pos, Ub), Lb);
                if t > T/2
                    new_pos = 0.5*(new_pos + Destination_position) + ...
                        search_radius.*randn(1,dim);
                end
                group(g).pos(i,:) = new_pos;
                
                % 反向学习策略
                if rand() < 0.3*E
                    opposite_pos = Ub + Lb - group(g).pos(i,:);
                    opposite_fit = fobj(opposite_pos);
                    if opposite_fit < group(g).fitness(i)
                        group(g).pos(i,:) = opposite_pos;
                    end
                end
                
                % 适应度计算
                group(g).fitness(i) = fobj(group(g).pos(i,:));
                
                % 更新个体最优
                if group(g).fitness(i) < group(g).pBestScore(i)
                    group(g).pBest(i,:) = group(g).pos(i,:);
                    group(g).pBestScore(i) = group(g).fitness(i);
                end
            end
            
            % 更新子群最优
            [min_fit, idx] = min(group(g).pBestScore);
            if min_fit < group(g).gBestScore
                group(g).gBestScore = min_fit;
                group(g).gBest = group(g).pBest(idx,:);
            end
        end
        
        % 更新全局最优和精英池
        for g = 1:groups
            if group(g).gBestScore < Destination_fitness
                Destination_fitness = group(g).gBestScore;
                Destination_position = group(g).gBest;
            end
        end
        
        % 维护精英池（保留多样化优质解）
        [~, sort_idx] = sort([elite_pool.fitness]);
        if Destination_fitness < elite_pool(sort_idx(end)).fitness
            elite_pool(sort_idx(end)) = struct('position',Destination_position,...
                                              'fitness',Destination_fitness);
        end
        
        last_fitness = Destination_fitness;
        Convergence_curve(t) = Destination_fitness;
    end

    Best_score = Destination_fitness;
    Best_pos = Destination_position;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Levy飞行函数
function [z] = levy(n, m, beta)
    num = gamma(1+beta) * sin(pi*beta/2);
    den = gamma((1+beta)/2) * beta * 2^((beta-1)/2);
    sigma_u = (num/den)^(1/beta);
    u = randn(n, m) * sigma_u;
    v = randn(n, m);
    z = u ./ abs(v).^(1/beta);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 归一化函数
function r = normr(x)
    if isnan(x) || isinf(x)
        r = 0;
    else
        r = min(1, x);
    end
end
