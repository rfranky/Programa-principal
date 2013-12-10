%Copyright (c) 2013   Ramon Franquesa Alberti, Carlos Martín Isla , Gonzalo Lopez Lillo , Aleix Gras Godoy 


clear all;
%%
types=struct;
types(1).type='concert';
types(2).type='conference';
types(3).type='exhibition';
types(4).type='fashion';
types(5).type='non_event';
types(6).type='other';
types(7).type='protest';
types(8).type='sports';
types(9).type='theater_dance';
ndeclasses=length(types);
flag=3;
tipusExtraccio='HISTBLOC';



%% LECTURA D'IMATGES, EXTRACCIÓ Y ESCRIPTURA DE MODELS

% saltar este paso si se dispone de modelosX.bin
tic
elementsperclasse=[];
dataset=[];
fid=fopen(strcat('modelos',tipusExtraccio,'.bin'),'w');
fwrite(fid,[]);
fclose(fid);
fid=fopen(strcat('modelos',tipusExtraccio,'.bin'),'a+');


for k=1:ndeclasses,
    
directori=strcat(types(k).type,'\');
model=lecturaImatges(directori,flag);

aux=dataset;
dataset=[aux ; model];

[m n]=size(model);
aux=elementsperclasse;
elementsperclasse=[aux  m];



end;

fwrite(fid,elementsperclasse,'uint16'); % PRIMERA FILA DE L'ARXIU : VECTOR AMB SEPARACIONS DE LA MATRIU DATASET
fwrite(fid,dataset, 'double'); % MATRIU AMB TOTS ELS MODELS, QUE ES DESENSAMBLA AMB EL VECTOR ANTERIOR
fclose(fid);
%%D'aquesta forma emns estalviem fer servir structs per models de mida variable, i aprofitem tota la
%%potència de matlab.
toc
%% RECUPERACIÓ DE MODELS


fid=fopen(strcat('modelos',tipusExtraccio,'.bin'),'r');
index=fread(fid,[1 ndeclasses],'uint16');
    
auxmat=fread(fid,[sum(index) 256*16],'double'); 




%% CLASSIFICACIÓ


archivo=fopen(strcat('resultats',tipusExtraccio,'.txt'),'w');
directori='clas/';
[nombre,n_model,H]=lecturaImatges_Aval(directori, flag);
[nil n]=size(n_model);%nil=numero de imagenes leidas
k=1;
for i=1:nil
    indice=classificador_knn(n_model(i,:),auxmat,index,k);
    types(indice).type;

    fprintf(archivo,strcat(nombre(i).noms));
    fprintf(archivo,' ');
    fprintf(archivo,types(indice).type);
    fprintf(archivo,'\r\n');
    
      if(mod(i,100)==0) %progreso
        x=num2str(floor(i*100/nil));
        display(strcat(x,'%'));end;
end
fclose(archivo);



%% AVALUACIÓ

x = csvimport('sed2013_task2_dataset_train_gs.csv');
% Comparem si és correcte o no la classificació.
asn1 = csvimport(strcat('resultats',tipusExtraccio,'.txt'));

n = length(x);
m = length(asn1);
IN=[1 2 3 4 5 6 7 8 9];
OUT=[NaN NaN NaN NaN NaN NaN NaN NaN NaN];

for k = 1:m
    [id clase_pre] = spacecuter(asn1(k));
    evento_pre = strcat(clase_pre);


for i = 2:n
[on clase_tro] = spacecuter(x(i));
ev =troba(id, on);
if (ev == length(id))
    evento = strcat(clase_tro);
    IN(k+9) = avaluat(evento_pre);
    OUT(k+9) = avaluat(evento);
      break;
end
end
end



g1 = [IN];
g2 = [OUT];
[C,order] = confusionmat(g1,g2)
Pr=0
Re=0
Fscore=0

%Avaluacio de les dades
for l = 1:9
PV=C(l,l);
PF=0;
NF=0;
for j = 1:9
   PF=PF+C(l,j);
   NF=NF+C(j,l);
end
  PF=PF-C(l,l);
  NF=NF-C(l,l);
Pr(l) = (PV/(PV+PF));
Re(l) = (PV/(PV+NF));
Fscore(l) = 2*((Pr(l)*Re(l))/(Pr(l)+Re(l)));

end

Pr_avg=mean(Pr);
Re_avg = mean(Re);
Fscore_avg = mean(Fscore);


f = figure('Position',[400 400 800 500]);
dat = C; 
cnames = {'Concert','Conference','Exhibition','Fashion','Non_event','Others','Protest','Sports','Theatre'};
rnames = {'Concert','Conference','Exhibition','Fashion','Non_event','Others','Protest','Sports','Theatre'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 200 700 200 ]);
set(t,'ColumnWidth',{50})

dat = [Pr]; 
rnames = {'Precision'};
cnames = {'Concert','Conference','Exhibition','Fashion','Non_event','Others','Protest','Sports','Theatre'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 0 700 50]);
set(t,'ColumnWidth',{50})

dat = [Re]; 
rnames = {'Recall'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 50 700 50]);
set(t,'ColumnWidth',{50})

dat = [Fscore];
rnames = {'F-score'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 100 700 50]);
set(t,'ColumnWidth',{50})

dat = [Pr_avg Re_avg Fscore_avg];
rnames = {'Avg'};
cnames = {'Precision', 'Recall','F-score'};
t = uitable('Parent',f,'Data',dat,'ColumnName',cnames,'RowName',rnames,'Position',[30 450 200 50]);
set(t,'ColumnWidth',{50})
%%
figure(3);
subplot(4,1,1);
plot(Pr);title('Precision');
axis([1 9 0 1]);
subplot(4,1,2);
plot(Re);title('Recall');
axis([1 9 0 1]);
subplot(4,1,3);
plot(Fscore);title('F-score');
axis([1 9 0 1]);
subplot(4,1,4);
plot(Pr,Re);title('Corba PR');ylabel('Precision');xlabel('Recall');
axis([0 1 0 1]);

%% Normal Mutual Information

 MIhat = MutualInfo(C,C);
