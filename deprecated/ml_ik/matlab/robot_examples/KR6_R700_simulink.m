%% KUKA KR6 R700 — Simulink / Simscape Multibody Visualization
% This script:
%   1. Imports the URDF into a Simulink model via smimport
%   2. Generates joint trajectory data (HOME → TARGET)
%   3. Saves the trajectory to workspace for Simulink to use
%   4. Opens the model so you can press "Run" to see the animation
%
% Prerequisites: Simscape Multibody, Simulink

clc; clear; close all;

%% 0. Paths
script_dir = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(script_dir);
addpath(fullfile(matlab_dir, 'automatic_IK'));
addpath(fullfile(matlab_dir, 'general-robotics-toolbox'));
addpath(fullfile(matlab_dir, 'rand_helpers'));
addpath(fullfile(matlab_dir, 'robot_IK_helpers'));
addpath(matlab_dir);

%% 1. Solve IK for the target pose
kin = hardcoded_IK_setups.KR6_R700.get_kin();

T_target = [0.600; 0.100; 0.400];
R_target = eye(3);

[Q, is_LS] = hardcoded_IK.KR6_R700(R_target, T_target);

% Pick the best solution
[e_all, ~, ~] = robot_IK_error(struct('kin',kin,'R',R_target,'T',T_target), struct('Q',Q));
[~, best_idx] = min(e_all);
q_goal = Q(:, best_idx);

fprintf('Target:  [%.3f, %.3f, %.3f]\n', T_target);
fprintf('Best IK: [');
fprintf('%.4f ', q_goal');
fprintf(']\n');

%% 2. Generate joint trajectory (HOME → TARGET)
q_home = zeros(6, 1);
T_sim = 3;           % total simulation time (seconds)
dt = 0.01;           % time step
t_vec = (0:dt:T_sim)';
N = length(t_vec);

% Smooth trajectory using a cosine profile (S-curve)
% s goes from 0 to 1 smoothly
s = 0.5 * (1 - cos(pi * t_vec / T_sim));

% Joint angle trajectories (N x 6)
q_traj = zeros(N, 6);
for j = 1:6
    q_traj(:, j) = q_home(j) + s * (q_goal(j) - q_home(j));
end

% Create timeseries objects for Simulink Signal Builder / From Workspace
for j = 1:6
    joint_ts{j} = timeseries(q_traj(:,j), t_vec); %#ok<SAGROW>
end

% Save to workspace as a struct for Simulink "From Workspace" blocks
joint_data.time = t_vec;
joint_data.signals.values = q_traj;
joint_data.signals.dimensions = 6;

% Also save individual joints for compatibility
for j = 1:6
    eval(sprintf('joint_%d_data = timeseries(q_traj(:,%d), t_vec);', j, j));
end

fprintf('Trajectory generated: %.1f seconds, %d steps\n', T_sim, N);

%% 3. Import URDF into Simulink
urdf_path = fullfile(matlab_dir, 'kr6_r700_2_clean.urdf');
model_name = 'KR6_R700_sim';

fprintf('Importing URDF into Simulink model "%s"...\n', model_name);

% Close existing model if open
if bdIsLoaded(model_name)
    close_system(model_name, 0);
end

% Import the URDF
try
    smimport(urdf_path, 'ModelName', model_name);
    fprintf('✓ Simulink model "%s" created successfully!\n', model_name);
catch e
    fprintf('smimport error: %s\n', e.message);
    fprintf('\nTrying with ModelSimplification...\n');
    try
        smimport(urdf_path, 'ModelName', model_name, ...
                 'ModelSimplification', 'groupRigidBodies');
        fprintf('✓ Simulink model "%s" created with simplified topology!\n', model_name);
    catch e2
        fprintf('Failed: %s\n', e2.message);
        return;
    end
end

%% 4. Set simulation parameters
set_param(model_name, 'StopTime', num2str(T_sim));
set_param(model_name, 'Solver', 'ode45');
set_param(model_name, 'MaxStep', '0.01');

fprintf('\n=== Setup Complete ===\n');
fprintf('Simulation time: %.1f seconds\n', T_sim);
fprintf('Joint trajectory data is in the workspace.\n');
fprintf('\nTo run the simulation:\n');
fprintf('  1. The model "%s" should now be open in Simulink\n', model_name);
fprintf('  2. Press the green "Run" button (or Ctrl+T)\n');
fprintf('  3. The Mechanics Explorer window will show the 3D animation\n');
fprintf('\nNote: The mesh files in the URDF reference a Linux path.\n');
fprintf('The robot will appear as primitive shapes (cylinders/boxes),\n');
fprintf('but the joint kinematics and animation will be correct.\n');

%% 5. Open the model
open_system(model_name);
fprintf('\nModel opened. Press Run to animate!\n');
