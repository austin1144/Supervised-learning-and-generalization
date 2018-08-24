clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

%generation of examples and targets
%x=0:0.01:3*pi; y=sin(x.^2)+rand(1);
x=0:0.05:3*pi; y=sin(abs(x))+rand(1);
% x=0:0.05:3*pi; y=sin(x.^2);
p=con2seq(x); t=con2seq(y); % convert the data to a useful format
neuron=50;
error =1e-5;
%creation of networks

net1=feedforwardnet(neuron,'traingda'); 
net2=feedforwardnet(neuron,'traincgf'); 
net3=feedforwardnet(neuron,'trainbfg'); 
net4=feedforwardnet(neuron,'trainlm'); 
 
net1.trainParam.goal = error;
net2.trainParam.goal = error;
net3.trainParam.goal = error;
net4.trainParam.goal = error;

    
    
% net2=feedforwardnet(50,'trainbr');  different method
net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};
net3.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net3.lw{2,1}=net1.lw{2,1};
net3.b{1}=net1.b{1};
net3.b{2}=net1.b{2};
net4.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net4.lw{2,1}=net1.lw{2,1};
net4.b{1}=net1.b{1};
net4.b{2}=net1.b{2};

net1.trainParam.epochs=1000;
net2.trainParam.epochs=1000;
net3.trainParam.epochs=1000;
net4.trainParam.epochs=1000;
%%%%aply four net
net1=train(net1,p,t);
net2=train(net2,p,t);
net3=train(net1,p,t);
net4=train(net2,p,t);
a13=sim(net1,p); a23=sim(net2,p);
a33=sim(net3,p); a43=sim(net4,p);

%plots
% figure
% subplot(3,3,1);
% plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g'); % plot the sine function and the output of the networks
% title('1 epoch');
% legend('target','trainlm','trainbfg','Location','north');
% subplot(3,3,2);
% postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
% subplot(3,3,3);
% postregm(cell2mat(a21),y);
% %
% subplot(3,3,4);
% % plot(x(1:30),y(1:30),'bx',x(1:30),cell2mat(a12(1:30)),'r',x,cell2mat(a22(1:30)),'g');
% plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g');
% title('50 epochs');
% legend('target','trainlm','trainbfg','Location','north');
% subplot(3,3,5);
% postregm(cell2mat(a12),y);
% subplot(3,3,6);
% postregm(cell2mat(a22),y);
% %
% subplot(3,3,7);
% plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
% title('1000 epochs');
% legend('target','trainlm','trainbfg','Location','north');
% subplot(3,3,8);
% postregm(cell2mat(a13),y);
% subplot(3,3,9);
% postregm(cell2mat(a23),y);
%

%%%%%%%%%%%%%%%%%
figure
end_point = 135;
% subplot(1,3,7);
plot(x(1:end_point),y(1:end_point),'bx',x(1:end_point),cell2mat(a13(1:end_point)),'r',x(1:end_point),cell2mat(a23(1:end_point)),'g',x(1:end_point),cell2mat(a33(1:end_point)),'c',x(1:end_point),cell2mat(a43(1:end_point)),'y');
title('1000 epochs');
legend('target','traingda','traincgf','trainbfg','trainlm','Location','north');