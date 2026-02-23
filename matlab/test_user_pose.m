clc; clear;

% Add paths robustly
script_dir = fileparts(mfilename('fullpath'));
if endsWith(script_dir, 'robot_examples')
    matlab_dir = fileparts(script_dir);
elseif endsWith(script_dir, 'matlab')
    matlab_dir = script_dir;
else
    matlab_dir = fullfile(pwd, 'matlab');
end

addpath(fullfile(matlab_dir, 'automatic_IK'));
addpath(fullfile(matlab_dir, 'general-robotics-toolbox'));
addpath(fullfile(matlab_dir, 'rand_helpers'));
addpath(fullfile(matlab_dir, 'robot_IK_helpers'));

% Get KUKA KR6 R700 Kinematics
kin = hardcoded_IK_setups.KR6_R700.get_kin();

% --- USER DEFINED POSE ---
% Position: [0.3, 0.4, 0.25]
T_target = [0.3; 0.4; 0.25];
% Rotation: Identity (no orientation specified, assuming pointing straight)
R_target = eye(3);

fprintf('==== VALIDATING IK-GEO FOR KUKA KR6 R700 ====\n\n');
fprintf('Target Position T: [%.3f, %.3f, %.3f]\n', T_target);
fprintf('Target Rotation R:\n');
disp(R_target);

%% 1. Solve Joint Angles strictly using IK-GEO
fprintf('--- Solving via IK-Geo ---\n');
[Q, is_LS] = hardcoded_IK.KR6_R700(R_target, T_target);

n_sols = size(Q, 2);
if n_sols == 0
    fprintf('No valid solutions found for this pose!\n');
    return;
else
    fprintf('IK-Geo found %d solutions.\n\n', n_sols);
end

%% 2. Validate all solutions using Forward Kinematics (FK)
fprintf('--- Validating All Solutions via Forward Kinematics ---\n');
fprintf('%5s  %13s  %13s  %13s\n', 'Sol #', 'Pos Error', 'Rot Error', 'Total Error');
fprintf('------------------------------------------------------\n');

for i = 1:n_sols
    q_sol = Q(:, i);
    
    % Forward Kinematics
    [R_check, T_check] = fwdkin(kin, q_sol);
    
    % Calculate Error
    e_T = norm(T_check - T_target);
    e_R = norm(R_check - R_target);
    e_total = e_T + e_R;
    
    fprintf('%5d  %13.2e  %13.2e  %13.2e\n', i, e_T, e_R, e_total);
end

fprintf('------------------------------------------------------\n');
fprintf('Note: Errors around 1e-15 or 1e-16 mean perfect exact mathematical precision.\n');
