%% KUKA KR6 R700 — Inverse Kinematics Demo
% Demonstrates IK-Geo solving for the KUKA KR6 R700
% Kinematic family: Spherical wrist + 2 parallel axes → IK_spherical_2_parallel
%
% Workflow: pose → IK → FK → verify

clc; clear;

% Add required paths
% Add required paths
% (assumes the script is run from within the ik-geo/matlab/robot_examples directory,
% or that its folder is added to path and we are in ik-geo/matlab)
matlab_dir = fileparts(pwd);
if ~endsWith(matlab_dir, 'matlab')
    matlab_dir = pwd;
end
addpath(fullfile(matlab_dir, 'automatic_IK'));
addpath(fullfile(matlab_dir, 'general-robotics-toolbox'));

addpath(fullfile(matlab_dir, 'rand_helpers'));
addpath(fullfile(matlab_dir, 'robot_IK_helpers'));
addpath(matlab_dir);

%% 1. Define kinematics
kin = hardcoded_IK_setups.KR6_R700.get_kin();

fprintf('=== KUKA KR6 R700 IK Demo ===\n\n');
fprintf('Joint axes H:\n'); disp(kin.H);
fprintf('Link offsets P:\n'); disp(kin.P);

%% 2. Verify kinematic family
fprintf('--- Axis Detection ---\n');
[is_int, is_int_nc, is_par, is_sph] = detect_intersecting_parallel_axes(kin);
print_intersecting_parallel_axes(is_int, is_int_nc, is_par, is_sph);
fprintf('\nRecommended solvers:\n');
rec_solver_6_DOF(is_int, is_int_nc, is_par, is_sph);

%% 3. Define a specific target pose
% Target: pos=[0.600, 0.100, 0.400], RPY=[0,0,0] → R=eye(3)
T_target = [0.600; 0.100; 0.400];
R_target = eye(3);  % quaternion [0,0,0,1] = identity

fprintf('\n--- Target Pose ---\n');
fprintf('Target position T: [%.3f, %.3f, %.3f]\n', T_target);
fprintf('Target rotation R (identity):\n');
disp(R_target);

%% 4. Solve IK — get ALL solutions
[Q, is_LS] = hardcoded_IK.KR6_R700(R_target, T_target);

fprintf('--- IK Solutions ---\n');
n_sols = size(Q, 2);
fprintf('Number of solutions found: %d\n\n', n_sols);

for i = 1:n_sols
    fprintf('Solution %d: [', i);
    fprintf('%.4f ', Q(:,i)');
    fprintf(']\n');
end

%% 5. Verify: FK each IK solution and compare to target
fprintf('\n--- Verification (FK of each IK solution) ---\n');
fprintf('%5s  %12s  %12s  %12s  %6s\n', 'Sol#', 'Pos Error', 'Rot Error', 'Total Error', 'is_LS');

for i = 1:n_sols
    [R_check, T_check] = fwdkin(kin, Q(:,i));
    e_T = norm(T_check - T_target);
    e_R = norm(R_check - R_target);
    e_total = e_R + e_T;

    % Check is_LS type (may be vector or matrix depending on solver)
    if numel(is_LS) >= i
        if size(is_LS, 1) > 1
            ls_flag = any(is_LS(:,i));
        else
            ls_flag = is_LS(i);
        end
    else
        ls_flag = false;
    end

    fprintf('%5d  %12.2e  %12.2e  %12.2e  %6s\n', ...
        i, e_T, e_R, e_total, string(ls_flag));
end

%% 6. Best solution
[e_all, e_R_all, e_T_all] = robot_IK_error(struct('kin',kin,'R',R_target,'T',T_target), struct('Q',Q));
[best_err, best_idx] = min(e_all);
fprintf('\nBest solution: #%d (error = %.2e)\n', best_idx, best_err);
fprintf('Best q: [');
fprintf('%.6f ', Q(:,best_idx)');
fprintf(']\n');

%% 7. Multi-trial correctness test
fprintf('\n--- Multi-Trial Correctness Test (100 trials) ---\n');
N_trials = 100;
max_best_err = 0;
n_failed = 0;

for trial = 1:N_trials
    q_rand = rand_angle([6 1]);
    [R_t, T_t] = fwdkin(kin, q_rand);
    [Q_t, ~] = hardcoded_IK.KR6_R700(R_t, T_t);

    if isempty(Q_t)
        n_failed = n_failed + 1;
        continue;
    end

    e_t = robot_IK_error(struct('kin',kin,'R',R_t,'T',T_t), struct('Q',Q_t));
    best_e = min(e_t);
    max_best_err = max(max_best_err, best_e);
end

fprintf('Trials: %d | Failed (no solution): %d\n', N_trials, n_failed);
fprintf('Max best-error across all trials: %.2e\n', max_best_err);

if max_best_err < 1e-6
    fprintf('✓ ALL TESTS PASSED\n');
else
    fprintf('✗ SOME TESTS HAD HIGH ERROR\n');
end

%% 8. Visualize the Robot Moving from Home to Target
fprintf('\n--- 3D Visualization ---\n');
try
    % Load URDF model
    urdf_path = fullfile(matlab_dir, 'kr6_r700_2.urdf');
    if ~isfile(urdf_path)
        urdf_path = fullfile(pwd, 'kr6_r700_2.urdf');
    end
    
    robot = importrobot(urdf_path);
    robot.DataFormat = 'column';
    
    % Define start (home) and end (IK solution) configurations
    q_home = zeros(6, 1);
    q_goal = Q(:, best_idx);
    
    % Get end-effector positions for markers
    [~, T_home_pos] = fwdkin(kin, q_home);
    [~, T_goal_pos] = fwdkin(kin, q_goal);
    
    % --- Step 1: Show home configuration ---
    fig = figure('Name', 'KUKA KR6 R700 — Home → Target', ...
                 'NumberTitle', 'off', 'Color', 'w', ...
                 'Position', [100 100 900 700]);
    
    show(robot, q_home, 'PreservePlot', false);
    hold on;
    
    % Mark home end-effector (blue) and target (red)
    plot3(T_home_pos(1), T_home_pos(2), T_home_pos(3), ...
          'bo', 'MarkerSize', 12, 'MarkerFaceColor', 'b', 'DisplayName', 'Home EE');
    plot3(T_goal_pos(1), T_goal_pos(2), T_goal_pos(3), ...
          'rp', 'MarkerSize', 18, 'MarkerFaceColor', 'r', 'DisplayName', 'Target');
    
    % Dashed line from home to target
    plot3([T_home_pos(1) T_goal_pos(1)], ...
          [T_home_pos(2) T_goal_pos(2)], ...
          [T_home_pos(3) T_goal_pos(3)], ...
          'k--', 'LineWidth', 1.5, 'DisplayName', 'Path');
    
    % Label the points
    text(T_home_pos(1), T_home_pos(2), T_home_pos(3)+0.06, 'HOME', ...
         'FontSize', 10, 'FontWeight', 'bold', 'Color', 'b', 'HorizontalAlignment', 'center');
    text(T_goal_pos(1), T_goal_pos(2), T_goal_pos(3)+0.06, 'TARGET', ...
         'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r', 'HorizontalAlignment', 'center');
    
    view(45, 30);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    title(sprintf('KUKA KR6 R700 | HOME config | Sol #%d', best_idx));
    legend('Location', 'northeast');
    drawnow;
    
    fprintf('Showing HOME configuration for 2 seconds...\n');
    pause(2);
    
    % --- Step 2: Animate from home to target ---
    fprintf('Animating from HOME to TARGET...\n');
    N_steps = 60;
    for i = 1:N_steps
        alpha = (i-1) / (N_steps-1);  % 0 → 1
        q_now = (1-alpha)*q_home + alpha*q_goal;
        
        show(robot, q_now, 'PreservePlot', false);
        title(sprintf('Moving... %.0f%%', alpha*100));
        drawnow;
    end
    
    % --- Step 3: Final pose with markers ---
    title(sprintf('TARGET REACHED | IK Solution #%d | Error: %.1e', best_idx, best_err));
    fprintf('Animation complete! Close the figure window to continue.\n');
    
    % Keep the figure open until the user closes it
    waitfor(fig);
    fprintf('Figure closed.\n');
    
catch e
    fprintf('Visualization error: %s\n', e.message);
    fprintf('Ensure you have the Robotics System Toolbox installed.\n');
end


