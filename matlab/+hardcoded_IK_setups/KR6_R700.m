classdef KR6_R700
% KUKA KR6 R700 — Spherical wrist + 2 parallel axes (h2 || h3)
% Kinematic family: IK_spherical_2_parallel (same as IRB 6640)
%
% POE parameters extracted from kr6_r700_2.urdf
% H(:,2) = H(:,3) = ey   → 2 parallel axes
% P(:,5), P(:,6) lie in span of adjacent axes → spherical wrist

methods (Static)
    function kin = get_kin()
        ex = [1;0;0];
        ey = [0;1;0];
        ez = [0;0;1];

        % Joint axes in base frame (at zero config)
        % Derived from URDF origin RPY chain:
        %   S_1=Rx(pi), S_2=Rx(-pi/2), S_3=S_2, 
        %   S_4=[0 0 -1;0 1 0;1 0 0], S_5=Rx(-pi/2), S_6=S_4
        %   H(:,i) = S_i * [0;0;1]
        kin.H = [-ez, ey, ey, -ex, ey, -ex];

        % Canonical representation (shifted origins to physical wrist center)
        % to make spherical wrist explicit (P(:,5)=0, P(:,6)=0)
        kin.P = [ ...
            [0; 0; 0.208], ...          % P(:,1)
            [0.025; 0.0907; 0.192], ... % P(:,2)
            [0.335; -0.0042; 0], ...    % P(:,3)
            [0.365; -0.0865; 0.025], ...% P(:,4) (original + 0.224 along X)
            [0; 0; 0], ...              % P(:,5)
            [0; 0; 0], ...              % P(:,6)
            [0.09; 0; 0] ...            % P(:,7) (original + 0.0615 along X)
        ];

        kin.joint_type = zeros([6 1]);
    end

    function [P, S] = setup()
        S.Q = rand_angle([6,1]);
        [P.R, P.T] = fwdkin(hardcoded_IK_setups.KR6_R700.get_kin(), S.Q);
    end

    function S = run(P)
        [S.Q, S.is_LS] = IK.IK_2_parallel(P.R, P.T, ...
            hardcoded_IK_setups.KR6_R700.get_kin());
    end

    function [e, e_R, e_T] = error(P, S)
        P.kin = hardcoded_IK_setups.KR6_R700.get_kin();
        [e, e_R, e_T] = robot_IK_error(P, S);
    end
end
end
