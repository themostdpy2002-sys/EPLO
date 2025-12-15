function [Best_score, Best_pos, Convergence_curve] = PLO(pop, T, lb, ub, dim, fobj)
    tic;

    % 参数合法性检查
    %valid_dims = [2, 10, 30, 50, 100]; % CEC2017 支持的维度
    %if ~ismember(dim, valid_dims)
    %    error(['Error: Test functions are only defined for D=', ...
    %           num2str(valid_dims)]);
    %end

    % 初始化参数
    pos = lb + (ub - lb) * rand(pop, dim); % 初始化位置
    vel = zeros(pop, dim);                 % 初始化速度
    fitness = inf(pop, 1);                 % 初始化适应度
    pBest = pos;                           % 粒子个体最优位置
    pBestScore = inf(pop, 1);              % 粒子个体最优分数
    gBestScore = inf;                      % 全局最优分数
    gBest = zeros(1, dim);                 % 全局最优位置

    % 收敛曲线
    max_iterations = T;
    Convergence_curve = inf(1, max_iterations); % Initialize with inf to ensure all entries are updated

    % 动态权重参数
    wMax = 0.9; 
    wMin = 0.4;
    c1 = 2.0; 
    c2 = 2.0;

    % 主循环
    for t = 1:max_iterations
        w = wMax - (wMax - wMin) * t / max_iterations; % 动态惯性权重

        for i = 1:pop
            % 更新速度和位置
            vel(i, :) = w * vel(i, :) + ...
                        c1 * rand(1, dim) .* (pBest(i, :) - pos(i, :)) + ...
                        c2 * rand(1, dim) .* (gBest - pos(i, :));
            pos(i, :) = pos(i, :) + vel(i, :);

            % 边界处理
            pos(i, :) = max(min(pos(i, :), ub), lb);

            % 计算适应度
            try
                fitness(i) = fobj(pos(i, :)); % 调用适应函数
            catch
                error('Error evaluating the fitness function. Check the input dimension or function compatibility.');
            end

            % 更新个体最优
            if fitness(i) < pBestScore(i)
                pBest(i, :) = pos(i, :);
                pBestScore(i) = fitness(i);
            end
        end

        % 更新全局最优
        [current_gBestScore, idx] = min(pBestScore);
        if current_gBestScore < gBestScore
            gBestScore = current_gBestScore;
            gBest = pBest(idx, :);
        end

        % 记录收敛曲线
        Convergence_curve(t) = gBestScore;

        % 显示进度
        if mod(t, 10) == 0
            disp(['Iteration ', num2str(t), ': Best Fitness = ', num2str(gBestScore)]);
        end
    end

    % 返回结果
    Best_pos = gBest;
    Best_score = gBestScore;

    toc;
end