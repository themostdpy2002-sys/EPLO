function cost = fitnessPenalized(path, threat, r, sx, sy, sz, ex, ey, ez, gridCnt)
    penalty = 1e6;  % ⬅ 大惩罚，禁止穿越威胁球
    N = gridCnt + 2;
    pts = zeros(N,3);
    pts(1,:) = [sx, sy, sz];
    for i = 1:gridCnt
        idx = 3*(i-1)+1;
        pts(i+1,:) = path(idx:idx+2);
    end
    pts(end,:) = [ex, ey, ez];

    cost = 0;
    for i = 1:N-1
        d = norm(pts(i+1,:) - pts(i,:));
        cost = cost + d;
        if segmentIntersects(pts(i,:), pts(i+1,:), threat, r)
            cost = cost + penalty;
        end
    end
end

function flag = segmentIntersects(A, B, threat, r)
    flag = false;
    for k = 1:size(threat,1)
        C = threat(k,:);
        R = r(k);
        v = B - A;
        t = dot(C - A, v) / dot(v,v);
        t = max(0, min(1, t));
        proj = A + t * v;
        if norm(C - proj) < R
            flag = true;
            return;
        end
    end
end
