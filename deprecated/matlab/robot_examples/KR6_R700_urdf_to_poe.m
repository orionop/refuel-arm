%% KUKA KR6 R700 — URDF to Product of Exponentials conversion
% Extracts H, P from kr6_r700_2.urdf joint data
% Then detects intersecting/parallel axes and recommends IK solvers
%
% Conversion formula:
%   S_i = R_origin_01 * R_origin_12 * ... * R_origin_{i-1,i}
%   H(:,i) = S_i * [0;0;1]
%   P(:,1) = p_origin_01
%   P(:,i+1) = S_i * p_origin_{i,i+1}

clc; clear;

%% URDF joint data (all axes are local [0;0;1])
% Each entry: {rpy, xyz}
joint_rpy = {
    [pi, 0, 0]            % joint_1
    [pi/2, 0, 0]          % joint_2
    [0, 0, 0]             % joint_3
    [pi/2, 0, -pi/2]      % joint_4
    [0, pi/2, pi/2]       % joint_5
    [pi/2, 0, -pi/2]      % joint_6
};

joint_xyz = {
    [0; 0; 0.208]                  % joint_1
    [0.025; -0.0907; -0.192]       % joint_2
    [0.335; 0; -0.0042]            % joint_3
    [0.141; -0.025; -0.0865]       % joint_4
    [0; 0.0505; -0.224]            % joint_5
    [0.0615; 0; -0.0505]           % joint_6
};

% Flange/tool0 (fixed joint)
tool_rpy = [pi, pi*1.2246467991473532E-16, pi];
tool_xyz = [0; 0; -0.0285];

n_joints = 6;
local_axis = [0; 0; 1];

%% Compute POE parameters
H = zeros(3, n_joints);
P = zeros(3, n_joints + 1);

% P(:,1) = first joint origin translation
P(:, 1) = joint_xyz{1};

% S_i = accumulated origin rotations
S = eye(3);

for i = 1:n_joints
    % Origin rotation: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R_origin = rpy_to_rotmat(joint_rpy{i});
    
    % Accumulated rotation up to joint i
    S = S * R_origin;
    
    % Joint axis in base frame
    H(:, i) = S * local_axis;
    H(:, i) = H(:, i) / norm(H(:, i));
    
    % Offset to next joint (or tool)
    if i < n_joints
        P(:, i + 1) = S * joint_xyz{i + 1};
    end
end

% Last offset P(:,6) uses joint 6 xyz (already handled above for i<6)
% Actually we need P(:,2) through P(:,7)
% Recompute properly:
P = zeros(3, n_joints + 1);
P(:, 1) = joint_xyz{1};

S_vec = cell(n_joints, 1);
S_accum = eye(3);
for i = 1:n_joints
    R_origin = rpy_to_rotmat(joint_rpy{i});
    S_accum = S_accum * R_origin;
    S_vec{i} = S_accum;
end

for i = 2:n_joints
    P(:, i) = S_vec{i-1} * joint_xyz{i};
end

% Tool offset
R_tool = rpy_to_rotmat(tool_rpy);
P(:, n_joints + 1) = S_vec{n_joints} * tool_xyz;

% Joint axes
for i = 1:n_joints
    H(:, i) = S_vec{i} * local_axis;
    H(:, i) = H(:, i) / norm(H(:, i));
end

%% Build kin structure
kin.H = H;
kin.P = P;
kin.joint_type = zeros(6, 1);

%% Display
fprintf('=== KUKA KR6 R700 — POE Parameters ===\n\n');
fprintf('H (joint axes):\n'); disp(H);
fprintf('P (link offsets):\n'); disp(P);

%% Detect axes and recommend solvers
fprintf('=== Axis Detection ===\n');
[is_int, is_int_nc, is_par, is_sph] = detect_intersecting_parallel_axes(kin);
print_intersecting_parallel_axes(is_int, is_int_nc, is_par, is_sph);
fprintf('\n');
rec_solver_6_DOF(is_int, is_int_nc, is_par, is_sph);

%% Quick FK test
fprintf('\n=== FK Test at q=0 ===\n');
[R0, T0] = fwdkin(kin, zeros(6,1));
fprintf('Position at q=0: [%.4f, %.4f, %.4f]\n', T0);
fprintf('Rotation at q=0:\n'); disp(R0);

%% RPY to rotation matrix (URDF convention: R = Rz(yaw)*Ry(pitch)*Rx(roll))
function R = rpy_to_rotmat(rpy)
    r = rpy(1); p = rpy(2); y = rpy(3);
    Rx = [1 0 0; 0 cos(r) -sin(r); 0 sin(r) cos(r)];
    Ry = [cos(p) 0 sin(p); 0 1 0; -sin(p) 0 cos(p)];
    Rz = [cos(y) -sin(y) 0; sin(y) cos(y) 0; 0 0 1];
    R = Rz * Ry * Rx;
end
