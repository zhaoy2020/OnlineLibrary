% 极坐标系 
% theta=linspace(0,2*pi,20);
% rho=5*theta/10;
% polarplot(theta,rho,'-o','LineWidth',2)
% rlim([0,4])


% 绘制玫瑰线
theta=0:0.02:(2*pi);
R=10;
k=5;
rho=R*sin(k*theta);
polarplot(theta,rho,'LineWidth',2)
rlim([0,2+R])
