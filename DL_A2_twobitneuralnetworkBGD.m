clc; 
clear all; 
close all;
%Bengie, Henry, Ketan
%Part A: 2-bit logic gates
rng(2)
inputs = [0 0 1; 0 1 1; 1 0 1; 1 1 1];
%output = [0, 1, 1, 1];
%output = [1, 0, 0, 0];
%output = [0, 0, 0, 1];
%output = [1, 1, 1, 0];
output = [0, 1, 1, 0];
input_nodes = 3;
hidden_nodes = 3;
output_nodes = 1;
W1 = rand(input_nodes,hidden_nodes);
W2 = rand(hidden_nodes,output_nodes);
Bias1 = rand(hidden_nodes,1);
Bias2 = rand(output_nodes,1);
rho = 0.1; %learning rate
max_iter = 35000;
error=zeros(output_nodes,max_iter);

for i = 1:max_iter
    
   %estimated output
        
        op_w=inputs*W1;
        
        op_sig=1./(1+exp(-(op_w)));
        
        p=op_sig*W2;
        
        out=1./(1+exp(-(p))); 
        e=-(output'-out);
        delta=(out.*(1-out)).*e;
       
        %hidden layer weights updation
        
        W2=W2-rho.*op_sig'*delta;
        delta_hid=op_sig.*(1-op_sig).*(delta*W2');
        
        %input layer weight updations
        
        W1=W1-rho*(inputs'*delta_hid);  

        op_w=inputs*W1;
        
        op_sig=1./(1+exp(-(op_w)));
        
        p=op_sig*W2;
        
        out=1./(1+exp(-(p)));

        e=-(output-out);
        
        % convergence verification
        if(norm(e)<0.5)
            break;
        end
end

testing_data = inputs;
sse=sum((error.^2),1);

    op_w=testing_data*W1;
    op_sig=1./(1+exp(-(op_w)));
    out=(1./(1+exp(-(op_sig*W2))));

disp("Convergence Output :")
disp(out)
disp("W1 :")
disp(W1)
disp("W2 :")
disp(W2)