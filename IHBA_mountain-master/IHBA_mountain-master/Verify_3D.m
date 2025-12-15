function result = Verify_3D(path, threats, radii, safeDist)
% 三维路径可行性验证函数
% 输入:
%   path - N×3矩阵，路径点坐标
%   threats - M×3矩阵，威胁中心坐标
%   radii - M×1向量，威胁半径
%   safeDist - 安全距离阈值
% 输出:
%   result - 包含验证结果的结构体

% 设置默认安全距离
if nargin < 4
    safeDist = 0;
end

% 初始化结果结构
result = struct(...
    'isValid', true, ...       % 路径是否有效
    'minDist', 0, ...          % 路径总长度
    'minDistToThreat', inf, ...% 到威胁的最小距离
    'bendCount', 0, ...        % 弯曲次数
    'bendAngles', [] ...       % 弯曲角度
);

%% 1. 计算路径总长度
n = size(path, 1);
for i = 1:n-1
    result.minDist = result.minDist + norm(path(i,:) - path(i+1,:));
end

%% 2. 威胁规避验证
numThreats = size(threats, 1);
minSurfaceDist = inf;  % 最小表面距离

% 检查每个线段与威胁的关系
for i = 1:n-1
    A = path(i, :);
    B = path(i+1, :);
    
    for k = 1:numThreats
        C = threats(k, :);
        R = radii(k) + safeDist;  % 扩大后的半径
        
        % 计算线段AB到威胁球心C的最短距离
        dist = distPointToSegment(C, A, B);
        
        % 检查是否侵入威胁区
        if dist < R
            result.isValid = false;
        end
        
        % 更新最小表面距离（实际距离，不考虑安全距离）
        d_actual = dist - radii(k);
        if d_actual < minSurfaceDist
            minSurfaceDist = d_actual;
        end
    end
end
result.minDistToThreat = minSurfaceDist;

%% 3. 曲线弯曲检测
angleThreshold = 5;  % 弯曲角度阈值(度)
angles = [];

if n >= 3
    for i = 2:n-1
        % 计算三个连续点形成的两个向量
        v1 = path(i-1, :) - path(i, :);
        v2 = path(i+1, :) - path(i, :);
        
        % 计算向量夹角
        cosTheta = dot(v1, v2) / (norm(v1) * norm(v2));
        cosTheta = max(min(cosTheta, 1), -1);  % 确保在有效范围内
        theta = acos(cosTheta) * 180/pi;       % 转换为角度
        
        % 只记录超过阈值的弯曲
        if theta > angleThreshold
            result.bendCount = result.bendCount + 1;
            angles(end+1) = theta;
        end
    end
end
result.bendAngles = angles;
end

%% 辅助函数：计算点到线段的最短距离
function dist = distPointToSegment(P, A, B)
% 计算点P到线段AB的最短距离
AP = P - A;
AB = B - A;

% 计算投影长度
lenAB = norm(AB);
if lenAB == 0  % A和B重合
    dist = norm(AP);
    return;
end

% 计算投影比例
t = dot(AP, AB) / (lenAB * lenAB);
t = max(0, min(1, t));  % 限制在[0,1]范围内

% 计算最近点
closest = A + t * AB;
dist = norm(P - closest);
end