addpath('general-robotics-toolbox');
addpath('general-robotics-toolbox/axis-detection');
addpath('automatic_IK');
addpath('robot_IK_helpers');
addpath('rand_helpers');

kin = hardcoded_IK_setups.KR6_R700.get_kin();
R = eye(3);
T = [0.600; 0.100; 0.400];

[Q_sph, is_LS_sph] = IK.IK_spherical_2_parallel(R, T, kin);
disp('Q_sph:'); disp(Q_sph);

disp('--- Validation for Q_sph ---');
for i = 1:size(Q_sph, 2)
    [R_fk, T_fk] = fwdkin(kin, Q_sph(:, i));
    e_T = norm(T_fk - T);
    e_R = norm(R_fk - R, 'fro');
    fprintf('Solution %d: Pos Error = %e, Rot Error = %e\n', i, e_T, e_R);
end
