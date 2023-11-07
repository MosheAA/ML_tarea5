 
function label = clasificador_gaussiano(x, gauss_model, P )
% Funcion para producir etiquetas de acuerdo al clasificador gaussiano
% gauss model= estructura procedente de la funcion
% clasificador_gaussiano_train
% P probabilidad a priori de la clase 1 
% x vector en Rn cuya etiqueta se quiere predecir 

P_M = P;
P_N = 1-P_M;

Q_M=gauss_model.QC1 ;
mu_x_M=gauss_model.MUC1 ;

Q_N=gauss_model.QC2 ;
mu_x_N= gauss_model.MUC2 ;


data_T = x;


n = size(data_T,2);

% vector de etiquetas predichas
label_pred = 9*ones(size(data_T,1),1);

for ob = 1:size(data_T,1)
    x= data_T(ob,:);

    E_CM = ([x-mu_x_M]/Q_M)*[x-mu_x_M]';
    E_CN = ([x-mu_x_N]/Q_N)*[x-mu_x_N]';


    p_CM = (1/sqrt(det(Q_M)))*(1/(2*pi).^(n/2))*exp(-(1/2)*E_CM)*P_M;
    p_CN = (1/sqrt(det(Q_N)))*(1/(2*pi).^(n/2))*exp(-(1/2)*E_CN)*P_N;

% ASIGNACION ETIQUETA REGLA MAP 
if p_CM>p_CN
label_pred(ob) = 1;
elseif p_CM<p_CN
label_pred(ob) = 0;    
else
label_pred(ob) =p_CM; 
end


end

label = label_pred;
end