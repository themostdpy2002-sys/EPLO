function [mapXY, threatPoints, threatRadii] = prepareThreats(csvFullPath)
    data = readtable(csvFullPath);

    % ======================== STEP 1: 经纬度 -> 平面坐标 ========================
    R = 6371000;
    latRad = deg2rad(data.latitude);
    lonRad = deg2rad(data.longitude);
    lon0   = mean(data.longitude);  % 中央经度
    mapX = R * (lonRad - deg2rad(lon0)) .* cos(latRad);
    mapY = R * latRad;
    mapXY = [mapX, mapY];

    % ======================== STEP 2: 数据清洗 & 筛选 ========================
    % 人为构造伪障碍（雷达测距小、风速大、但未标 obstacle）
    pseudoObstacleIdx = find(data.lidar_distance < 60 & data.wind_speed > 5);
    data.obstacle_detected(pseudoObstacleIdx) = 1;

    % 只取本地集中区域内的点（防止离群）
    xMid = mean(mapX); yMid = mean(mapY);
    maxRadius = 400; % 限制区域半径（单位米）
    distanceToCenter = vecnorm([mapX - xMid, mapY - yMid], 2, 2);
    validIdx = distanceToCenter < maxRadius;

    % 最终障碍点提取
    obsIdx = find(data.obstacle_detected == 1 & validIdx);
    obsXY = mapXY(obsIdx, :);

    if size(obsXY, 1) < 10
        error('有效障碍点太少，请检查条件。');
    end

    % ======================== STEP 3: 聚类建模威胁区域 ========================
    epsVal = 25; minPts = 4;  % DBSCAN参数
    labels = dbscan(obsXY, epsVal, minPts);

    uniqueLabels = unique(labels(labels > 0));
    threatPoints = zeros(numel(uniqueLabels), 2);
    threatRadii  = zeros(numel(uniqueLabels), 1);

    for i = 1:numel(uniqueLabels)
        pts = obsXY(labels == uniqueLabels(i), :);
        center = mean(pts, 1);
        radius = max(vecnorm(pts - center, 2, 2));
        threatPoints(i,:) = center;
        threatRadii(i) = radius * 1.2;  % 安全放缩
    end
end
