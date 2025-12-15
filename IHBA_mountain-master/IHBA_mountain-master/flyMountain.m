%% flyMountain_multiUAV_dynamic.m
% Multi-path Multi-UAV planning, path conflict detection, dynamic obstacle avoidance (integrated PPLO)
clear; clc; close all;

% Set default figure and axes background to white
set(0, 'DefaultFigureColor', [1 1 1]);
set(0, 'DefaultAxesColor', [1 1 1]);

% --- Increase default font sizes for all plots and texts ---
set(0, 'DefaultAxesFontSize', 14, 'DefaultTextFontSize', 14);
set(0, 'DefaultLegendFontSize', 14);

%% 1. Environment parameters - Load threat zones from CSV (optimized clustering)
fprintf('Loading threat data from CSV...\n');
data = readtable('data/dynamic_obstacle_dataset_full (1).csv');

% Extract detected obstacle points
obstacleData = data(data.obstacle_detected == 1, :);
fprintf('Detected %d obstacle points\n', height(obstacleData));

% Coordinate conversion parameters
R = 6371000; % Earth radius (meters)
deg2rad = pi/180;
lat0 = mean(obstacleData.latitude);
lon0 = mean(obstacleData.longitude);

% Convert to Cartesian coordinates (East, North, Alt)
x = (obstacleData.longitude - lon0) .* (R * cos(lat0*deg2rad) * deg2rad);
y = (obstacleData.latitude  - lat0) .* (R * deg2rad);
z = obstacleData.altitude * 0.01;  % altitude scaling
points = [x, y, z];

% Data normalization
meanVals = mean(points);
stdVals = std(points);
points = (points - meanVals) ./ stdVals;

% --- Optimized DBSCAN clustering parameters ---
eps = 0.35;      % neighborhood radius for clustering
minPts = 3;      % minimum core points

fprintf('Using DBSCAN clustering: eps=%.2f, minPts=%d\n', eps, minPts);
labels = dbscan(points, eps, minPts);

% Process clustering results
uniqueLabels = unique(labels);
uniqueLabels(uniqueLabels == -1) = []; % remove noise label
threat = [];
r = [];
threat_ids = {}; % store threat zone IDs

fprintf('Clustering result: %d threat zones\n', length(uniqueLabels));

for i = 1:length(uniqueLabels)
    clusterIdx = (labels == uniqueLabels(i));
    clusterPoints = points(clusterIdx, :);
    
    % Restore original scale
    rawPoints = clusterPoints .* stdVals + meanVals;
    
    % Compute cluster center (consider XY plane only)
    center = mean(rawPoints(:,1:2), 1);
    
    % Compute cluster radius: max distance + safety margin
    dists = sqrt(sum((rawPoints(:,1:2) - center).^2, 2));
    radius = max(dists) + 15;  % 15 m safety buffer
    
    threat = [threat; center];
    r = [r; radius];
    threat_ids{end+1} = sprintf('T%d', i); % assign threat ID
    
    fprintf('Threat %s: center(%.1f, %.1f), radius %.1f m, contains %d points\n', ...
            threat_ids{end}, center(1), center(2), radius, sum(clusterIdx));
end

% Add noise points as small threat zones
noiseIdx = (labels == -1);
if any(noiseIdx)
    noisePts = points(noiseIdx, :);
    rawNoise = noisePts .* stdVals + meanVals;
    for i = 1:size(rawNoise,1)
        threat = [threat; rawNoise(i,1:2)];
        r = [r; 10];  % fixed small radius (m)
        threat_ids{end+1} = sprintf('N%d', i); % noise threat ID
        fprintf('Added small noise threat point: center(%.1f, %.1f), radius 10 m\n', ...
                rawNoise(i,1), rawNoise(i,2));
    end
end

% Threat zone merging logic
fprintf('Merging contained smaller threat zones...\n');
% Sort by radius descending
[r_sorted, sortIdx] = sort(r, 'descend');
threat = threat(sortIdx, :);
r = r(sortIdx);
threat_ids = threat_ids(sortIdx);

% Mark threats to remove if fully contained within a larger threat
toRemove = false(size(r));
for i = 1:length(r)
    if toRemove(i), continue; end % skip already marked
    for j = i+1:length(r)
        if toRemove(j), continue; end
        % center distance
        distCenters = norm(threat(i,:) - threat(j,:));
        % if larger threat fully contains smaller one
        if distCenters + r(j) <= r(i)
            fprintf('Threat %s (r=%.1f) fully contains threat %s (r=%.1f), removing %s\n', ...
                    threat_ids{i}, r(i), threat_ids{j}, r(j), threat_ids{j});
            toRemove(j) = true;
        end
    end
end

% Delete marked threats
threat(toRemove, :) = [];
r(toRemove) = [];
threat_ids(toRemove) = [];
fprintf('Number of threat zones after merge: %d\n', size(threat,1));

% Ensure a reasonable number of threat zones (5-8)
minThreats = 5;
maxThreats = 8;
if size(threat,1) < minThreats
    fprintf('Not enough threat zones (<%d), adding extra random threats...\n', minThreats);
    numToAdd = minThreats - size(threat,1);
    for i = 1:numToAdd
        % Randomly generate threats within data range
        xRange = [min(x), max(x)];
        yRange = [min(y), max(y)];
        
        randX = xRange(1) + (xRange(2)-xRange(1)) * rand();
        randY = yRange(1) + (yRange(2)-yRange(1)) * rand();
        randR = 25 + 20 * rand(); % radius 25-45 m
        
        threat = [threat; randX, randY];
        r = [r; randR];
        threat_ids{end+1} = sprintf('R%d', i); % random threat ID
        fprintf('Added random threat %s: center(%.1f, %.1f), radius %.1f m\n', ...
                threat_ids{end}, randX, randY, randR);
    end
elseif size(threat,1) > maxThreats
    fprintf('Too many threat zones (%d), randomly selecting %d...\n', size(threat,1), maxThreats);
    randIdx = randperm(size(threat,1), maxThreats);
    threat = threat(randIdx, :);
    r = r(randIdx);
    threat_ids = threat_ids(randIdx);
end

%% 2. Dynamic threat zone parameters
fprintf('Initializing dynamic threat zone parameters...\n');
numThreats = size(threat, 1);
threat_speeds = 0.5 + 1.5 * rand(numThreats, 1); % speeds (0.5-2.0 m/s)
threat_directions = 2 * pi * rand(numThreats, 1); % random headings (0-2��)

% Movement boundaries for threats
boundary_margin = 50; % margin
x_min = min(x) - boundary_margin;
x_max = max(x) + boundary_margin;
y_min = min(y) - boundary_margin;
y_max = max(y) + boundary_margin;

% Threat types (0 - static, 1 - dynamic)
threat_types = ones(numThreats, 1); % default all dynamic
% Set ~30% to static
static_threats = randperm(numThreats, max(1, round(0.3*numThreats)));
threat_types(static_threats) = 0;

fprintf('Dynamic threat configuration:\n');
fprintf('ID\tType\tSpeed(m/s)\tDirection(rad)\tRadius(m)\n');
for i = 1:numThreats
    type_str = 'Dynamic';
    if threat_types(i) == 0
        type_str = 'Static';
    end
    fprintf('%s\t%s\t%.2f\t\t%.2f\t\t%.1f\n', ...
            threat_ids{i}, type_str, threat_speeds(i), threat_directions(i), r(i));
end

%% 3. Preset multiple path start/end points and UAV counts
% Expand start/end ranges by threat extents
xAll = [x; threat(:,1)];
yAll = [y; threat(:,2)];
xRange = [min(xAll), max(xAll)];
yRange = [min(yAll), max(yAll)];

% Three paths: start and end (x,y,z)
pathsStart = [
    xRange(1) + 0.1*(xRange(2)-xRange(1)), yRange(1) + 0.1*(yRange(2)-yRange(1)), 0;
    xRange(1) + 0.3*(xRange(2)-xRange(1)), yRange(1) + 0.7*(yRange(2)-yRange(1)), 0;
    xRange(1) + 0.8*(xRange(2)-xRange(1)), yRange(1) + 0.2*(yRange(2)-yRange(1)), 0
];

pathsEnd = [
    xRange(1) + 0.9*(xRange(2)-xRange(1)), yRange(1) + 0.9*(yRange(2)-yRange(1)), 100;
    xRange(1) + 0.7*(xRange(2)-xRange(1)), yRange(1) + 0.3*(yRange(2)-yRange(1)), 80;
    xRange(1) + 0.2*(xRange(2)-xRange(1)), yRange(1) + 0.8*(yRange(2)-yRange(1)), 90
];

% Ensure start/end safety
fprintf('Checking safety of start and end points...\n');
for pIdx = 1:size(pathsStart,1)
    % Check start
    startSafe = false;
    while ~startSafe
        startSafe = true;
        for k = 1:size(threat,1)
            dist = norm(pathsStart(pIdx,1:2) - threat(k,:));
            if dist < r(k) + 15 % safety buffer
                fprintf('Path %d start is inside threat %s (distance=%.2f < safe radius %.2f), regenerating...\n', ...
                        pIdx, threat_ids{k}, dist, r(k)+15);
                startSafe = false;
                % regenerate start
                pathsStart(pIdx,1:2) = [xRange(1) + rand()*(xRange(2)-xRange(1)), ...
                                       yRange(1) + rand()*(yRange(2)-yRange(1))];
                break;
            end
        end
    end
    
    % Check end
    endSafe = false;
    while ~endSafe
        endSafe = true;
        for k = 1:size(threat,1)
            dist = norm(pathsEnd(pIdx,1:2) - threat(k,:));
            if dist < r(k) + 15 % safety buffer
                fprintf('Path %d end is inside threat %s (distance=%.2f < safe radius %.2f), regenerating...\n', ...
                        pIdx, threat_ids{k}, dist, r(k)+15);
                endSafe = false;
                % regenerate end
                pathsEnd(pIdx,1:2) = [xRange(1) + rand()*(xRange(2)-xRange(1)), ...
                                     yRange(1) + rand()*(yRange(2)-yRange(1))];
                break;
            end
        end
    end
end
fprintf('All start and end points are safe!\n');

uavCountPerPath = [3, 2, 1]; % number of UAVs per path
numPaths = size(pathsStart,1);
totalUAVs = sum(uavCountPerPath);

%% 4. Generate each path (supports dynamic threats) and enforce threat avoidance
allRoutePts = cell(numPaths,1);
allSmoothPaths = cell(numPaths,1);
pathLengths = zeros(numPaths,1); % store path lengths
safetyMargin = 5; % m, minimum margin outside threat radius

fprintf('Generating initial paths...\n');
for pIdx = 1:numPaths
    % Generate tangent points for obstacle avoidance - limit number of points
    tangentPts = generateTangentPoints(threat, r);
    
    % Limit maximum nodes to avoid explosion
    maxNodes = 50;
    if size(tangentPts, 1) > maxNodes
        fprintf('Warning: path %d has %d tangent points, sampling %d randomly\n', pIdx, size(tangentPts,1), maxNodes);
        randIdx = randperm(size(tangentPts,1), maxNodes);
        tangentPts = tangentPts(randIdx, :);
    end
    
    nodes = [pathsStart(pIdx,:); tangentPts; pathsEnd(pIdx,:)];
    N = size(nodes,1);
    
    % Build graph: connect node pairs that do not intersect threats
    G = inf(N);
    for i = 1:N-1
        for j = i+1:N
            if ~segmentIntersects(nodes(i,:), nodes(j,:), threat, r)
                G(i,j) = norm(nodes(i,:)-nodes(j,:));
            end
        end
    end

    % Dijkstra shortest path
    [dist, prev] = dijkstra_shortest(G,1,N);
    
    % Reconstruct path
    if isinf(dist)
        fprintf('Warning: path %d has no feasible path, using straight connection\n', pIdx);
        routePts = [pathsStart(pIdx,:); pathsEnd(pIdx,:)];
    else
        route = [N];
        current = N;
        maxSteps = N*2;
        stepCount = 0;
        pathValid = true;
        
        while current ~= 1 && stepCount < maxSteps
            stepCount = stepCount + 1;
            prevNode = prev(current);
            
            % Validate predecessor
            if prevNode < 1 || prevNode > N || prevNode == 0
                fprintf('Invalid predecessor node %d -> %d, using straight connection\n', current, prevNode);
                pathValid = false;
                break;
            end
            
            route = [prevNode, route];
            current = prevNode;
        end
        
        if ~pathValid || stepCount >= maxSteps
            fprintf('Path reconstruction failed, using straight connection\n');
            routePts = [pathsStart(pIdx,:); pathsEnd(pIdx,:)];
        else
            routePts = nodes(route,:);
        end
    end
    
    allRoutePts{pIdx} = routePts;

    % B-spline smoothing (may cross threats; enforce correction later)
    M = size(routePts,1);
    if M < 2
        smoothPath = routePts;
    else
        t = 1:M;
        tq = linspace(1,M,100);
        xs = spline(t, routePts(:,1).', tq);
        ys = spline(t, routePts(:,2).', tq);
        zs = spline(t, routePts(:,3).', tq);
        smoothPath = [xs.', ys.', zs.'];
    end
    
    % Enforce threat avoidance (XY plane) and do light smoothing
    smoothPath = enforceThreatAvoidanceOnPath(smoothPath, threat, r, safetyMargin);
    
    allSmoothPaths{pIdx} = smoothPath;
    
    % Compute path length
    pathLengths(pIdx) = calculatePathLength(smoothPath);
    fprintf('Path %d initial length (after correction): %.2f m\n', pIdx, pathLengths(pIdx));
end

%% 5. Assign each UAV a path
allUAVPaths = cell(totalUAVs,1);
uavIdxGlobal = 0;
for pIdx = 1:numPaths
    nUAVs = uavCountPerPath(pIdx);
    for i = 1:nUAVs
        uavIdxGlobal = uavIdxGlobal + 1;
        allUAVPaths{uavIdxGlobal} = allSmoothPaths{pIdx};
    end
end

%% 6. Path conflict detection
safeDist = 15;
% Use each path's own frame count, but for scheduling use common min frames (see evaluateSchedule)
numFrames = size(allUAVPaths{1},1);
conflictMat = false(totalUAVs);
conflictFrameMat = cell(totalUAVs);

fprintf('Checking path conflicts...\n');
for i = 1:totalUAVs-1
    for j = i+1:totalUAVs
        % If path lengths differ, use minimum common frames
        nf_i = size(allUAVPaths{i},1);
        nf_j = size(allUAVPaths{j},1);
        nf = min(nf_i, nf_j);
        distVec = vecnorm(allUAVPaths{i}(1:nf,:) - allUAVPaths{j}(1:nf,:), 2, 2);
        conflictFrames = find(distVec < safeDist);
        if ~isempty(conflictFrames)
            conflictMat(i,j) = true;
            conflictMat(j,i) = true;
            conflictFrameMat{i,j} = conflictFrames;
            conflictFrameMat{j,i} = conflictFrames;
            fprintf('UAV%d and UAV%d have conflicts at %d frames\n', i, j, length(conflictFrames));
        end
    end
end

%% 7. Takeoff scheduling optimization using PPLO strategy
fprintf('Optimizing takeoff schedule (PPLO)...\n');
% Initialize delays
uavDelays = zeros(totalUAVs,1);

% Optimization parameters
popSize = 20;       % population size
maxIter = 500;      % max iterations
lb = 0;             % min delay
ub = 200;           % max delay
dim = totalUAVs;    % optimization dimension

% Objective: minimize max completion time + conflict penalties
fitnessFunc = @(delays) evaluateSchedule(delays, allUAVPaths, conflictFrameMat, numFrames, safeDist);

% Run PPLO (record convergence curve)
[bestDelays, bestFitness, convCurve] = PPLO(popSize, maxIter, lb, ub, dim, fitnessFunc);

fprintf('Optimization complete! Best fitness: %.2f\n', bestFitness);
uavDelays = round(bestDelays); % integer-frame delays

%% 8. Visualization
figure(1); clf;
set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);
set(gcf, 'Color', [1 1 1]);  % ensure white figure background
ax = subplot(2,2,[1,3]);
set(ax, 'Position', [0.05, 0.1, 0.65, 0.85]);
set(ax, 'Color', [1 1 1]);   % white axes background

hold(ax, 'on');
grid(ax, 'on');
axis(ax, 'equal');
rotate3d(ax, 'on');
xlabel(ax, 'X (m)', 'FontSize', 16); ylabel(ax, 'Y (m)', 'FontSize', 16); zlabel(ax, 'Z (m)', 'FontSize', 16);
view(ax, 3);
title(ax, sprintf('Multi-path Multi-UAV Obstacle Avoidance Planning (%d threat zones)', size(threat,1)), 'FontSize', 18);

% Colored threat zones
[xsph, ysph, zsph] = sphere(20);
threatColors = jet(size(threat,1)); % jet colormap
threatHandles = gobjects(numThreats, 1); % graphic handles for threats
threatTextHandles = gobjects(numThreats, 1); % text handles for threats

for i = 1:size(threat,1)
    % Draw 3D threat sphere
    threatHandles(i) = surf(ax, threat(i,1)+r(i)*xsph, threat(i,2)+r(i)*ysph, r(i)*zsph,...
        'EdgeColor','none', 'FaceAlpha',0.3, 'FaceColor',threatColors(i,:));
    
    % Label threat ID (larger font)
    threatTextHandles(i) = text(ax, threat(i,1), threat(i,2), r(i)+5, threat_ids{i}, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 14, ...
        'BackgroundColor', [1,1,1]);
end

% Draw original waypoint routes
colors = lines(numPaths);
for pIdx = 1:numPaths
    plot3(ax, allRoutePts{pIdx}(:,1), allRoutePts{pIdx}(:,2), allRoutePts{pIdx}(:,3), ...
        'o-', 'LineWidth', 1.5, 'Color', colors(pIdx,:), ...
        'MarkerFaceColor', colors(pIdx,:), 'MarkerSize', 6, ...
        'DisplayName', sprintf('Path %d original waypoints', pIdx));
end

% Draw smoothed paths and UAVs
uavColors = jet(totalUAVs);
uavIdxGlobal = 0;
for pIdx = 1:numPaths
    nUAVs = uavCountPerPath(pIdx);
    for i = 1:nUAVs
        uavIdxGlobal = uavIdxGlobal + 1;
        % Smoothed trajectory
        plot3(ax, allSmoothPaths{pIdx}(:,1), allSmoothPaths{pIdx}(:,2), allSmoothPaths{pIdx}(:,3),...
            '--', 'LineWidth', 1.5, 'Color', uavColors(uavIdxGlobal,:), ...
            'DisplayName', sprintf('UAV %d smoothed trajectory', uavIdxGlobal));
        
        % Start marker
        startP = pathsStart(pIdx,:);
        plot3(ax, startP(1), startP(2), startP(3), 's', 'MarkerSize',12, ...
            'MarkerFaceColor', uavColors(uavIdxGlobal,:), ...
            'MarkerEdgeColor', 'k', ...
            'DisplayName', sprintf('UAV %d start', uavIdxGlobal));
        
        % Show delay (larger font)
        text(ax, startP(1)+5, startP(2)+5, startP(3)+5, ...
            sprintf('Delay: %d', uavDelays(uavIdxGlobal)), ...
            'FontSize', 12, 'Color', uavColors(uavIdxGlobal,:), ...
            'BackgroundColor', [1,1,1]);
    end
end

% Mark goal points
for pIdx = 1:numPaths
    endP = pathsEnd(pIdx,:);
    plot3(ax, endP(1), endP(2), endP(3), 'p', 'MarkerSize',15, ...
        'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'Goal');
end

% --- Create legend that is split into 4 rows ---
% Collect plotted objects with DisplayName in plotting order
children = get(ax, 'Children');            % children: topmost first
children = flipud(children);               % flip to plotting order (first plotted first)
hasName = arrayfun(@(h) ~isempty(get(h,'DisplayName')) && ~strcmp(get(h,'DisplayName'),''), children);
handles = children(hasName);
labels  = arrayfun(@(h) get(h,'DisplayName'), handles, 'UniformOutput', false);
numEntries = numel(handles);
if numEntries == 0
    % fallback legend (nothing to show)
    lg = legend(ax, 'off');
else
    % compute number of columns so that legend has 4 rows
    numRows = 6;
    numCols = max(1, ceil(numEntries / numRows));
    lg = legend(ax, handles, labels, 'Location', 'northeastoutside', ...
                'NumColumns', numCols, 'FontSize', 14, 'Box', 'off', 'Interpreter', 'none');
end

xlim(ax, [min(x)-50, max(x)+50]);
ylim(ax, [min(y)-50, max(y)+50]);

% Plot convergence curve
figure(2); clf;
set(gcf, 'Units', 'normalized', 'Position', [0.15, 0.15, 0.7, 0.7]);
set(gcf, 'Color', [1 1 1]);
ax2 = subplot(1,1,1);
set(ax2, 'Position', [0.1, 0.1, 0.8, 0.8]);
set(ax2, 'Color', [1 1 1]);

hold(ax2, 'on');
grid(ax2, 'on');
plot(ax2, convCurve, 'b-', 'LineWidth', 1.8, 'MarkerSize', 4);
xlabel(ax2, 'Iterations', 'FontSize', 14);
ylabel(ax2, 'Best fitness', 'FontSize', 14);
title(ax2, 'PPLO Convergence Curve', 'FontSize', 16);
set(ax2, 'YScale', 'log', 'FontSize', 14);
text(ax2, 0.6, 0.9, sprintf('Final fitness: %.2f', bestFitness), ...
     'Units', 'normalized', 'FontSize', 12, 'BackgroundColor', [1,1,1]);

% Plot delay distribution
figure(3); clf;
set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);
set(gcf, 'Color', [1 1 1]);
ax3 = subplot(1,1,1);
set(ax3, 'Position', [0.15, 0.15, 0.75, 0.75]);
set(ax3, 'Color', [1 1 1]);

hold(ax3, 'on');
grid(ax3, 'on');
bar(ax3, uavDelays, 'FaceColor', [0.5, 0.8, 0.9]);
xlabel(ax3, 'UAV Index', 'FontSize', 14);
ylabel(ax3, 'Delay (frames)', 'FontSize', 14);
title(ax3, 'UAV Takeoff Delay Distribution', 'FontSize', 16);
set(ax3, 'FontSize', 14);

for i = 1:totalUAVs
    text(ax3, i, uavDelays(i), num2str(uavDelays(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'BackgroundColor', [1,1,1], 'FontSize', 12);
end

%% 9. Enhanced animation demo (supports dynamic threats)
fprintf('Preparing animation demo...\n');
figure(4); clf;
set(gcf, 'Position', [100, 100, 1200, 900], 'Name', 'Multi-UAV Dynamic Obstacle Avoidance Demo');
set(gcf, 'Color', [1 1 1]);
ax4 = axes;
hold(ax4, 'on'); grid(ax4, 'on'); axis(ax4, 'equal'); rotate3d(ax4, 'on');
set(ax4, 'Color', [1 1 1]);
xlabel(ax4, 'X (m)', 'FontSize', 16); ylabel(ax4, 'Y (m)', 'FontSize', 16); zlabel(ax4, 'Z (m)', 'FontSize', 16); 
view(ax4, 3);
title(ax4, sprintf('Multi-UAV Dynamic Obstacle Avoidance Demo (%d threats, %.0f%% dynamic)', size(threat,1), 100*mean(threat_types)), 'FontSize', 18);
set(ax4, 'FontSize', 14);

% Draw boundary rectangle
plot3(ax4, [x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], [0,0,0,0,0], '--k', 'LineWidth', 1.5);

% Create dynamic threat graphic objects
dynamicThreatHandles = gobjects(numThreats, 1);
dynamicThreatTextHandles = gobjects(numThreats, 1);
for i = 1:numThreats
    if threat_types(i) == 1
        % Dynamic threat - red
        dynamicThreatHandles(i) = surf(ax4, threat(i,1)+r(i)*xsph, threat(i,2)+r(i)*ysph, r(i)*zsph,...
            'EdgeColor','none', 'FaceAlpha',0.4, 'FaceColor', [1, 0.3, 0.3]);
    else
        % Static threat - blue
        dynamicThreatHandles(i) = surf(ax4, threat(i,1)+r(i)*xsph, threat(i,2)+r(i)*ysph, r(i)*zsph,...
            'EdgeColor','none', 'FaceAlpha',0.3, 'FaceColor', [0.3, 0.3, 1]);
    end
    
    % Label threat ID (larger font)
    dynamicThreatTextHandles(i) = text(ax4, threat(i,1), threat(i,2), r(i)+5, threat_ids{i}, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 14, ...
        'Color', [0,0,0], 'BackgroundColor', [1,1,1]);
end

% Draw planned paths (as background)
for pIdx = 1:numPaths
    plot3(ax4, allSmoothPaths{pIdx}(:,1), allSmoothPaths{pIdx}(:,2), allSmoothPaths{pIdx}(:,3),...
        '--', 'LineWidth', 1, 'Color', [0.7, 0.7, 0.7]);
end

% Create UAV graphics
hUAVs = gobjects(totalUAVs,1);
hTexts = gobjects(totalUAVs,1);
uavPaths = cell(totalUAVs,1); % store actual flight paths

for i = 1:totalUAVs
    % UAV marker
    hUAVs(i) = plot3(ax4, nan, nan, nan, ...
        'o', 'MarkerSize', 10, 'MarkerFaceColor', uavColors(i,:), ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    % UAV label (larger font)
    hTexts(i) = text(ax4, nan, nan, nan, sprintf('UAV%d', i), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'BackgroundColor', [1,1,1]);
    
    uavPaths{i} = []; % initialize
end

% Animation parameters
maxDelay = max(uavDelays);
totalFrames = numFrames + maxDelay + 150; % extra buffer
delayTime = 0.03; % pause per frame (s)
frameRate = 10; % Hz
dt = 1/frameRate; % time step (s)

% Real-time threat detection parameters
threatDetectionRange = 100; % m
replanInterval = 20; % frames

% Initialize threat positions
currentThreatPos = threat; % initial positions

% Initialize replan counters
replanCounters = zeros(totalUAVs,1);

% Initialize UAV states
uavStates = zeros(totalUAVs,1); % 0: not launched, 1: flying, 2: arrived
currentUAVPos = zeros(totalUAVs,3); % current positions
for i = 1:totalUAVs
    p = uavIndexToPath(i, uavCountPerPath);
    currentUAVPos(i,:) = pathsStart(p,:);
end

% Animation loop
for frameIdx = 1:totalFrames
    % ========== Update dynamic threat positions ==========
    for i = 1:numThreats
        if threat_types(i) == 1 % only update dynamic threats
            % compute new position
            dx = threat_speeds(i) * cos(threat_directions(i)) * dt;
            dy = threat_speeds(i) * sin(threat_directions(i)) * dt;
            newPos = currentThreatPos(i,:) + [dx, dy];
            
            % boundary check and bounce
            if newPos(1) < x_min || newPos(1) > x_max
                threat_directions(i) = pi - threat_directions(i); % horizontal bounce
                dx = threat_speeds(i) * cos(threat_directions(i)) * dt;
                newPos(1) = currentThreatPos(i,1) + dx;
            end
            if newPos(2) < y_min || newPos(2) > y_max
                threat_directions(i) = -threat_directions(i); % vertical bounce
                dy = threat_speeds(i) * sin(threat_directions(i)) * dt;
                newPos(2) = currentThreatPos(i,2) + dy;
            end
            
            % update position
            currentThreatPos(i,:) = newPos;
            
            % update graphics
            set(dynamicThreatHandles(i), 'XData', newPos(1)+r(i)*xsph, ...
                                         'YData', newPos(2)+r(i)*ysph);
            set(dynamicThreatTextHandles(i), 'Position', [newPos(1), newPos(2), r(i)+5]);
        end
    end
    
    % ========== Update UAV positions ==========
    for uavIdx = 1:totalUAVs
        % increment replan counter if flying
        if uavStates(uavIdx) == 1
            replanCounters(uavIdx) = replanCounters(uavIdx) + 1;
        end
        
        idx = frameIdx - uavDelays(uavIdx);
        
        if idx < 1
            % not launched yet
            p = uavIndexToPath(uavIdx, uavCountPerPath);
            pos = pathsStart(p,:);
            uavStates(uavIdx) = 0;
        elseif idx > size(allUAVPaths{uavIdx},1)
            % arrived at goal
            pos = allUAVPaths{uavIdx}(end,:);
            uavStates(uavIdx) = 2;
        else
            % flying
            uavStates(uavIdx) = 1;
            pos = allUAVPaths{uavIdx}(round(idx),:);
            
            % ========== Real-time threat detection ==========
            for tIdx = 1:numThreats
                % distance to threat (use current threat position)
                distToThreat = norm(pos(1:2) - currentThreatPos(tIdx,:));
                
                % if inside threat
                if distToThreat < r(tIdx) + 10 % 10 m safety buffer
                    fprintf('Warning: UAV%d at frame %d entered threat %s (distance=%.2fm)\n', ...
                            uavIdx, frameIdx, threat_ids{tIdx}, distToThreat);
                    % emergency avoidance - climb vertically
                    pos(3) = pos(3) + 20; % climb 20 m
                end
                
                % trigger replanning (simple strategy)
                if distToThreat < threatDetectionRange && replanCounters(uavIdx) >= replanInterval
                    fprintf('UAV%d at frame %d detected threat %s (distance=%.2fm), triggering simple replan\n', ...
                            uavIdx, frameIdx, threat_ids{tIdx}, distToThreat);
                    % simplified: vertical climb evasive maneuver
                    pos(3) = pos(3) + 15;
                    replanCounters(uavIdx) = 0;
                end
            end
        end
        
        % Update UAV position
        currentUAVPos(uavIdx,:) = pos;
        set(hUAVs(uavIdx), 'XData', pos(1), 'YData', pos(2), 'ZData', pos(3));
        
        % Update UAV label position
        set(hTexts(uavIdx), 'Position', [pos(1), pos(2), pos(3)+10]);
        
        % Record path
        uavPaths{uavIdx} = [uavPaths{uavIdx}; pos];
    end
    
    % Display current frame info (larger title font)
    title(ax4, sprintf('Multi-UAV Dynamic Avoidance - Frame: %d/%d (Dynamic: %d / Static: %d)', ...
          frameIdx, totalFrames, sum(threat_types), sum(~threat_types)), 'FontSize', 16);
    
    % Draw threat movement directions
    if mod(frameIdx, 5) == 0 % update direction every 5 frames
        delete(findobj(ax4, 'Tag', 'ThreatDirection'));
        for i = 1:numThreats
            if threat_types(i) == 1
                % direction arrow
                endX = currentThreatPos(i,1) + 0.8*r(i)*cos(threat_directions(i));
                endY = currentThreatPos(i,2) + 0.8*r(i)*sin(threat_directions(i));
                quiver(ax4, currentThreatPos(i,1), currentThreatPos(i,2), ...
                      endX - currentThreatPos(i,1), endY - currentThreatPos(i,2), ...
                      'AutoScale', 'off', 'LineWidth', 2, ...
                      'Color', 'r', 'MaxHeadSize', 1.5, 'Tag', 'ThreatDirection');
            end
        end
    end
    
    drawnow;
    
    % Pause control
    if delayTime > 0
        pause(delayTime);
    end
end

%% 10. Performance analysis
% Compute flight times for each UAV
flightTimes = zeros(totalUAVs,1);
for i = 1:totalUAVs
    flightTimes(i) = size(allUAVPaths{i},1) + uavDelays(i);
end

% Display results
fprintf('\n======= Performance Analysis =======\n');
fprintf('UAV | Delay(frames) | TotalTime(frames) | PathLength(m) | ReplanCount\n');
fprintf('------------------------------------------------------------------\n');

uavIdxGlobal = 0;
for pIdx = 1:numPaths
    nUAVs = uavCountPerPath(pIdx);
    pathLength = calculatePathLength(allSmoothPaths{pIdx});
    
    for i = 1:nUAVs
        uavIdxGlobal = uavIdxGlobal + 1;
        if isempty(uavPaths{uavIdxGlobal})
            replanCount = 0;
        else
            replanCount = sum( diff( uavPaths{uavIdxGlobal}(:,3) ) > 5 );  % rough estimate of replan count
        end
        
        fprintf('%4d   | %8d      | %12d       | %11.2f  | %10d\n', ...
            uavIdxGlobal, uavDelays(uavIdxGlobal), flightTimes(uavIdxGlobal), ...
            pathLength, replanCount);
    end
end

fprintf('------------------------------------------------------------------\n');
fprintf('Max completion time: %.2f frames\n', max(flightTimes));
fprintf('Average completion time: %.2f frames\n', mean(flightTimes));
fprintf('Total delay frames: %.2f frames\n', sum(uavDelays));
fprintf('Detected %d threat zones (%d dynamic, %d static)\n', size(threat,1), sum(threat_types), sum(~threat_types));

%% --- Helper functions ---
function pts = generateTangentPoints(threat, r)
    % generate tangent-like sampling points around each threat (for waypoint candidates)
    delta = 20;
    n = size(threat,1);
    pts = zeros(n*8,3);
    idx = 1;
    angles = 0:pi/4:2*pi-pi/4;
    for k = 1:size(threat,1)
        C = [threat(k,:), 0];
        R = r(k) + delta;
        for a = angles
            pts(idx,:)   = C + [R*cos(a), R*sin(a), 0];
            idx = idx + 1;
        end
    end
    pts = pts(1:idx-1,:);
end

function flag = segmentIntersects(A, B, threat, r)
    % Check whether segment AB intersects any threat disk (in XY plane)
    flag = false;
    v = B - A;
    len = norm(v);
    if len == 0
        return;
    end
    v = v / len;
    for k = 1:size(threat,1)
        C = [threat(k,:), 0];
        R = r(k);
        w = C - A;
        proj = dot(w, v);
        if proj <= 0
            dist = norm(C - A);
        elseif proj >= len
            dist = norm(C - B);
        else
            perp = w - proj*v;
            dist = norm(perp);
        end
        if dist < R
            flag = true;
            return;
        end
    end
end

function [dist, prev] = dijkstra_shortest(G, s, t)
    % Simple Dijkstra implementation on adjacency matrix G
    n = size(G,1);
    dist = inf(1,n); 
    dist(s) = 0;
    prev = zeros(1,n);
    visited = false(1,n);
    for i = 1:n
        minDist = inf; u = -1;
        for j = 1:n
            if ~visited(j) && dist(j) < minDist
                minDist = dist(j); u = j;
            end
        end
        if u == -1, break; end
        if u == t, break; end
        visited(u) = true;
        for v = 1:n
            if ~visited(v) && G(u,v) < inf
                alt = dist(u) + G(u,v);
                if alt < dist(v)
                    dist(v) = alt;
                    prev(v) = u;
                end
            end
        end
    end
end

function length = calculatePathLength(path)
    length = 0;
    for i = 1:size(path,1)-1
        length = length + norm(path(i,:) - path(i+1,:));
    end
end

function fitness = evaluateSchedule(delays, allUAVPaths, conflictFrameMat, numFrames, safeDist)
    % Evaluate schedule: delays in frames for each UAV
    delays = max(0, round(delays));
    totalUAVs = numel(delays);
    flightTimes = zeros(totalUAVs,1);
    for i = 1:totalUAVs
        flightTimes(i) = numFrames + delays(i);
    end
    maxTime = max(flightTimes);
    conflictPenalty = 0;
    for i = 1:totalUAVs-1
        for j = i+1:totalUAVs
            if ~isempty(conflictFrameMat{i,j})
                startI = delays(i);
                endI = delays(i) + numFrames;
                startJ = delays(j);
                endJ = delays(j) + numFrames;
                overlapStart = max(startI, startJ);
                overlapEnd = min(endI, endJ);
                if overlapStart < overlapEnd
                    for t = overlapStart:overlapEnd
                        frameI = t - delays(i);
                        frameJ = t - delays(j);
                        if frameI >= 1 && frameI <= numFrames && frameJ >= 1 && frameJ <= numFrames
                            posI = allUAVPaths{i}(round(frameI),:);
                            posJ = allUAVPaths{j}(round(frameJ),:);
                            dist = norm(posI - posJ);
                            if dist < safeDist
                                conflictPenalty = conflictPenalty + 1000 * (safeDist - dist);
                            end
                        end
                    end
                end
            end
        end
    end
    fitness = maxTime + conflictPenalty;
end

function cost = optimizePathCost(delta, pathPts, threat, r)
    % cost function for path perturbation optimization (not used directly in main)
    delta = reshape(delta, [], 3);
    newPts = pathPts + delta(1:size(pathPts,1),:);
    pathLength = 0;
    for i = 1:size(newPts,1)-1
        pathLength = pathLength + norm(newPts(i,:) - newPts(i+1,:));
    end
    safetyPenalty = 0;
    for i = 1:size(newPts,1)-1
        if segmentIntersects(newPts(i,:), newPts(i+1,:), threat, r)
            safetyPenalty = safetyPenalty + 1000;
        end
    end
    smoothPenalty = 0;
    for i = 2:size(newPts,1)-1
        v1 = newPts(i,:) - newPts(i-1,:);
        v2 = newPts(i+1,:) - newPts(i,:);
        denom = norm(v1)*norm(v2);
        if denom == 0
            continue;
        end
        cosval = dot(v1,v2) / denom;
        cosval = max(-1, min(1, cosval));
        angle = acos(cosval);
        smoothPenalty = smoothPenalty + abs(angle - pi/2);
    end
    cost = pathLength + safetyPenalty + 0.1*smoothPenalty;
end

function pIdx = uavIndexToPath(uavIdx, uavCountPerPath)
    cumCounts = cumsum(uavCountPerPath(:).');
    pIdx = find(uavIdx <= cumCounts, 1, 'first');
    if isempty(pIdx)
        error('uavIndexToPath: cannot find path for UAV %d (check uavCountPerPath)', uavIdx);
    end
end

%% Enforce threat avoidance on a path (XY plane)
function newPath = enforceThreatAvoidanceOnPath(path, threat, r, margin)
    % path: Nx3
    % threat: Mx2
    % r: Mx1
    if isempty(path)
        newPath = path; return;
    end
    newPath = path;
    N = size(path,1);
    maxIter = 5;
    for it = 1:maxIter
        violated = false;
        for i = 2:N-1 % keep endpoints unchanged
            pt = newPath(i,1:2);
            for k = 1:size(threat,1)
                dist = norm(pt - threat(k,:));
                thresh = r(k) + margin;
                if dist < thresh
                    violated = true;
                    dir = pt - threat(k,:);
                    if norm(dir) < 1e-6
                        % random direction if coincident
                        ang = rand*2*pi;
                        dir = [cos(ang), sin(ang)];
                    else
                        dir = dir / norm(dir);
                    end
                    new2d = threat(k,:) + (thresh) * dir;
                    newPath(i,1:2) = new2d;
                    % keep altitude unchanged
                end
            end
        end
        % Light moving-average smoothing (do not change endpoints)
        for i = 2:N-1
            newPath(i,:) = 0.5*newPath(i,:) + 0.25*(newPath(i-1,:) + newPath(i+1,:));
        end
        if ~violated
            break;
        end
    end
    % Final check: project any remaining violating points onto nearest threat boundary
    for i = 2:N-1
        pt = newPath(i,1:2);
        for k = 1:size(threat,1)
            dist = norm(pt - threat(k,:));
            thresh = r(k) + margin;
            if dist < thresh
                dir = pt - threat(k,:);
                if norm(dir) < 1e-6
                    ang = rand*2*pi; dir = [cos(ang), sin(ang)];
                else
                    dir = dir / norm(dir);
                end
                newPath(i,1:2) = threat(k,:) + thresh * dir;
            end
        end
    end
end

%% --- PPLO function (parallel subpopulations) ---
function [Best_pos, Bestscore, Convergence_curve] = PPLO(N, MaxFEs, lb, ub, dim, fhd)
    groups = 4;
    Np     = max(2, floor(N/groups));
    wMax   = 0.9; wMin = 0.2;
    c1Max  = 2.5; c1Min = 0.5;
    c2Max  = 2.5; c2Min = 0.5;
    elite_pool_size = 5;
    elite_sigma = 0.01 * (ub - lb);
    migration_period = 20;
    elite_pool = repmat(struct('pos',[],'fit',inf),1,elite_pool_size);
    init = lb + (ub-lb).*rand(N,dim);
    for g = 1:groups
        idx = (g-1)*Np + (1:Np);
        idx = idx(idx <= size(init,1));
        group(g).pos        = init(idx,:);
        group(g).vel        = zeros(size(idx,2),dim);
        group(g).fitness    = inf(size(idx,2),1);
        group(g).pBest      = group(g).pos;
        group(g).pBestScore = inf(size(idx,2),1);
        group(g).gBest      = zeros(1,dim);
        group(g).gBestScore = inf;
    end
    Convergence_curve = zeros(1,MaxFEs);
    global_best_score  = inf;
    global_best_pos    = zeros(1,dim);
    for t = 1:MaxFEs
        tau = t/MaxFEs;
        w  = wMax - (wMax - wMin) * tau^1.5;
        c1 = c1Max - (c1Max - c1Min) * tau;
        c2 = c2Min + (c2Max - c2Min) * tau;
        pOpp = 0.5 * (1 - tau);
        for g = 1:length(group)
            Npg = size(group(g).pos,1);
            for i = 1:Npg
                group(g).vel(i,:) = w*group(g).vel(i,:) ...
                    + c1*rand(1,dim).*(group(g).pBest(i,:) - group(g).pos(i,:)) ...
                    + c2*rand(1,dim).*(group(g).gBest      - group(g).pos(i,:));
                group(g).pos(i,:) = group(g).pos(i,:) + group(g).vel(i,:);
                group(g).pos(i,:) = max(min(group(g).pos(i,:), ub), lb);
                if rand < pOpp
                    opp = ub + lb - group(g).pos(i,:);
                    opp = max(min(opp, ub), lb);
                    if fhd(opp) < fhd(group(g).pos(i,:))
                        group(g).pos(i,:) = opp;
                    end
                end
                fi = fhd(group(g).pos(i,:));
                group(g).fitness(i) = fi;
                if fi < group(g).pBestScore(i)
                    group(g).pBest(i,:)    = group(g).pos(i,:);
                    group(g).pBestScore(i) = fi;
                end
            end
            [fg, idx] = min(group(g).pBestScore);
            if fg < group(g).gBestScore
                group(g).gBestScore = fg;
                group(g).gBest      = group(g).pBest(idx,:);
            end
        end
        if mod(t, migration_period) == 0
            for g = 1:length(group)
                next_g = mod(g, length(group)) + 1;
                [~, worst_idx] = max(group(next_g).pBestScore);
                group(next_g).pos(worst_idx,:)       = group(g).gBest;
                group(next_g).pBest(worst_idx,:)     = group(g).gBest;
                group(next_g).pBestScore(worst_idx)  = group(g).gBestScore;
            end
        end
        for g = 1:length(group)
            cand_pos = group(g).gBest;
            cand_fit = group(g).gBestScore;
            [~, worst] = max([elite_pool.fit]);
            if cand_fit < elite_pool(worst).fit
                elite_pool(worst).pos = cand_pos;
                elite_pool(worst).fit = cand_fit;
            end
            if cand_fit < global_best_score
                global_best_score = cand_fit;
                global_best_pos   = cand_pos;
            end
        end
        if t>0.4*MaxFEs && mod(t,10)==0
            delta = levy(1,dim,1.5).*(ub-lb);
            cand = global_best_pos + delta;
            cand = max(min(cand,ub),lb);
            fc = fhd(cand);
            if fc < global_best_score
                global_best_score = fc;
                global_best_pos   = cand;
            end
        end
        if t>0.6*MaxFEs
            base = fhd(global_best_pos);
            eps  = 1e-6; grad = zeros(1,dim);
            for j=1:dim
                tmp=global_best_pos; tmp(j)=tmp(j)+eps;
                grad(j)=(fhd(tmp)-base)/eps;
            end
            if norm(grad)>0
                dir = -grad/norm(grad);
                cand = global_best_pos + 0.05*(ub-lb).*dir;
                cand = max(min(cand,ub),lb);
                fc = fhd(cand);
                if fc<global_best_score
                    global_best_score = fc;
                    global_best_pos   = cand;
                end
            end
        end
        if mod(t,50)==0
            noise = randn(1,dim).*elite_sigma;
            cand = global_best_pos + noise;
            cand = max(min(cand,ub),lb);
            fc = fhd(cand);
            if fc<global_best_score
                global_best_score = fc;
                global_best_pos   = cand;
            end
        end
        Convergence_curve(t) = global_best_score;
        if mod(t,100)==0
            fprintf('Iter %3d/%3d, Best = %.4e\n', t, MaxFEs, global_best_score);
        end
    end
    Best_pos  = global_best_pos;
    Bestscore = global_best_score;
end

%% Levy flight function
function z = levy(n, m, beta)
    num = gamma(1+beta)*sin(pi*beta/2);
    den = gamma((1+beta)/2)*beta*2^((beta-1)/2);
    sigma = (num/den)^(1/beta);
    u = randn(n,m)*sigma;
    v = randn(n,m);
    z = u ./ abs(v).^(1/beta);
end

%% DBSCAN implementation (kept)
function labels = dbscan(X, epsilon, MinPts)
    n = size(X, 1);
    labels = zeros(n, 1);
    clusterCount = 0;
    % 用原生 MATLAB 代码计算欧氏距离矩阵，替代 pdist2
    D = sqrt(sum((X - permute(X, [3 2 1])).^2, 2));
    D = squeeze(D);
    for i = 1:n
        if labels(i) ~= 0
            continue;
        end
        neighbors = find(D(i, :) <= epsilon);
        if numel(neighbors) < MinPts
            labels(i) = -1;
            continue;
        end
        clusterCount = clusterCount + 1;
        labels(i) = clusterCount;
        neighbors(neighbors == i) = [];
        seedSet = neighbors(:);
        j = 1;
        while j <= numel(seedSet)
            idx = seedSet(j);
            if labels(idx) == -1
                labels(idx) = clusterCount;
            end
            if labels(idx) ~= 0
                j = j + 1;
                continue;
            end
            labels(idx) = clusterCount;
            newNeighbors = find(D(idx, :) <= epsilon);
            if numel(newNeighbors) >= MinPts
                for k = 1:numel(newNeighbors)
                    if labels(newNeighbors(k)) == 0
                        seedSet = [seedSet; newNeighbors(k)];
                    end
                end
            end
            j = j + 1;
        end
    end
end
