clc; clear; close all;

% Variables simbólicas
syms x y z O r b real

% Sistema
f1 = O*(y-x);
f2 = x*(r - z) - y;
f3 = x*y - b*z;

% Resolver sistema
soluciones = solve([f1 == 0, f2 == 0, f3 == 0], [x, y, z]);

% Mostrar resultados simbólicos
disp('Puntos de equilibrio (simbólicos):')
disp([soluciones.x, soluciones.y, soluciones.z])