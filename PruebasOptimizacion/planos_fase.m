%% =========================================================
%  PLANOS DE FASE - Modelo de Lorenz
%  Sistema 2D y Atractor de Lorenz 3D
%% =========================================================

clear; clc; close all;

%% =========================================================
%  PARTE 1: PLANO DE FASE - SISTEMA 2D
%  dx/dt = x - y      →  Nulcline dx=0: y = x  (familia 1)
%  dy/dt = x^2 - 1    →  Nulcline dy=0: x = 1, x = -1  (familia 2)
%% =========================================================

figure('Name','Plano de Fase - Sistema 2D','NumberTitle','off', ...
       'Position',[100 100 750 620]);

[X, Y] = meshgrid(-3:0.3:3, -3:0.3:3);
dX = X - Y;
dY = X.^2 - 1;

% Campo vectorial normalizado
N  = sqrt(dX.^2 + dY.^2); N(N==0) = 1;
quiver(X, Y, dX./N, dY./N, 0.5, 'Color',[0.55 0.75 1], 'LineWidth',0.8);
hold on;

% --- Familia 1: Nulcline dx/dt = 0 → y = x ---
x_lin = linspace(-3, 3, 400);
h1 = plot(x_lin, x_lin, 'r-', 'LineWidth', 2.2, ...
          'DisplayName', 'Familia 1: y = x  (\dot{x}=0)');

% --- Familia 2: Nulcline dy/dt = 0 → x = 1  y  x = -1 ---
h2a = xline( 1, 'Color',[0.1 0.7 0.1], 'LineWidth', 2.2, ...
             'DisplayName', 'Familia 2: x = \pm1  (\dot{y}=0)');
xline(-1, 'Color',[0.1 0.7 0.1], 'LineWidth', 2.2, ...
      'HandleVisibility','off');

% Puntos de equilibrio: intersecciones → (1,1) y (-1,-1)
eq = [1 1; -1 -1];
h3 = plot(eq(:,1), eq(:,2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor','k', ...
          'DisplayName', 'Puntos de equilibrio');
text( 1.1,  1.2,  'E_1 = (1,1)',   'FontSize',10, 'Color','k');
text(-1.7, -1.2,  'E_2 = (-1,-1)', 'FontSize',10, 'Color','k');

% Trayectorias
tspan = [0 15];
opts  = odeset('RelTol',1e-8,'AbsTol',1e-10);
f2d   = @(t,s) [s(1)-s(2); s(1)^2-1];
CI    = [2.5, 2; -2.5,-2; 0.5, 2.5; -0.5,-2.5; 2,-2.5; -2, 2.5; 0,2; 0,-2];
col   = lines(size(CI,1));

for i = 1:size(CI,1)
    [~,sol] = ode45(f2d, tspan, CI(i,:), opts);
    mask = sol(:,1)>=-3 & sol(:,1)<=3 & sol(:,2)>=-3 & sol(:,2)<=3;
    if i == 1
        plot(sol(mask,1), sol(mask,2), 'Color', col(i,:), ...
             'LineWidth',1.3, 'DisplayName','Trayectorias');
    else
        plot(sol(mask,1), sol(mask,2), 'Color', col(i,:), ...
             'LineWidth',1.3, 'HandleVisibility','off');
    end
end

axis([-3 3 -3 3]);
xlabel('x', 'FontSize',13); ylabel('y', 'FontSize',13);
title('Plano de Fase — Sistema No Lineal 2D', 'FontSize',14, 'FontWeight','bold');
legend([h1, h2a, h3], 'Location','northwest', 'FontSize',10);
grid on; box on;


%% =========================================================
%  PARTE 2: ATRACTOR DE LORENZ 3D
%% =========================================================

sigma = 10; r = 28; b = 8/3;
lorenz = @(t,s) [sigma*(s(2)-s(1));
                 r*s(1) - s(2) - s(1)*s(3);
                 s(1)*s(2) - b*s(3)];

figure('Name','Atractor de Lorenz 3D','NumberTitle','off', ...
       'Position',[870 100 820 680]);

% Trayectorias del atractor
tspan3 = [0 60];
opts3  = odeset('RelTol',1e-9,'AbsTol',1e-11);
CI3    = [0.1,0,0; -0.1,0,0; 0.1,0.1,0; 1,1,1; -1,-1,1];
colores3 = {'#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261'};

for i = 1:size(CI3,1)
    [~,sol] = ode45(lorenz, tspan3, CI3(i,:), opts3);
    plot3(sol(:,1), sol(:,2), sol(:,3), ...
          'Color', colores3{i}, 'LineWidth', 0.8, ...
          'HandleVisibility','off');
    hold on;
end

% Puntos de equilibrio
bval  = sqrt(b*(r-1));
Eeq   = [0,0,0; bval,bval,r-1; -bval,-bval,r-1];
etiq  = {'E_0 = (0,0,0)', ...
         sprintf('E_+ = (%.1f, %.1f, %d)', bval,  bval,  r-1), ...
         sprintf('E_- = (%.1f, %.1f, %d)', -bval, -bval, r-1)};
col_eq = {'k','r','b'};

for i = 1:3
    scatter3(Eeq(i,1), Eeq(i,2), Eeq(i,3), 90, col_eq{i}, 'filled', ...
             'DisplayName', etiq{i});
end

xlabel('x — velocidad fluido', 'FontSize',11);
ylabel('y — temp. horizontal',  'FontSize',11);
zlabel('z — temp. vertical',    'FontSize',11);
title({'Atractor de Lorenz — Espacio de Fase 3D', ...
       ['\sigma=',num2str(sigma),'   r=',num2str(r),'   b=',num2str(b,'%.4f')]}, ...
      'FontSize',13,'FontWeight','bold');
legend('Location','northeast','FontSize',9);
grid on; box on;
view(30, 25);

fprintf('\n✓ Figuras generadas.\n');