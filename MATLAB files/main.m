clc;
clear all;
close all;

matrixName= ["32x24", "48x36", "96x72", "128x96", "160x120", "200x150"];

errorMatlab = zeros (200,length(matrixName));
errorGramMatrixSVD = zeros (200,length(matrixName));
errorSequential = zeros (200,length(matrixName));
errorSequentialCublas = zeros (200,length(matrixName));
errorParallelSemiShared = zeros (200,length(matrixName));
errorParallelGlobal = zeros (200,length(matrixName));
errorParallelShared = zeros (200,length(matrixName));

matrixDim = zeros (length(matrixName),2);
timeMatlabOneSided = zeros (length(matrixName),1);
timeMatlabSVD = zeros (length(matrixName),1);
timeHost = zeros (length(matrixName),1);
timeDeviceGramMatrixSVD = zeros (length(matrixName),1);
timeDeviceSequential = zeros (length(matrixName),1);
timeDeviceSequentialCublas = zeros (length(matrixName),1);
timeDeviceParallelSemiShared = zeros (length(matrixName),1);
timeDeviceParallelGlobal = zeros (length(matrixName),1);
timeDeviceParallelShared = zeros (length(matrixName),1);
for i = 1: length(matrixName)
    A = load ("../Matrix/A"+matrixName(i));
    matrixDim (i, :) = size(A);
    cols (i) = matrixDim (i, 2);
    tStart = tic;   %start timer
    singularValuesMatlab = One_Sided_Jacobi_Rotation(A);
    timeMatlabOneSided(i) = toc(tStart)*1000; %stop timer
    tStart = tic; %start timer
    [U,S,V] = svd(A);
    timeMatlabSVD(i) = toc(tStart)*1000; %stop timer
    fileID = fopen("../SingularValues/Matlab/Singular Values Matlab "+matrixName(i)+".txt", 'w');
    fprintf (fileID, '%f\n', singularValuesMatlab);
    errorMatlab(1: cols(i), i) = abs(diag(S) - singularValuesMatlab);
    %% load singular values files
    fileID = fopen("../SingularValues/CudaHost/Singular Values Cuda Host "+matrixName(i)+".txt", 'r');
    singularValuesHost = fscanf (fileID, '%f\n');
    
    fileID = fopen("../SingularValues/CudaDevice/GramMatrixSVD/Singular Values Cuda Device "+matrixName(i)+".txt", 'r');
    singularValuesGramMatrixSVD = fscanf (fileID, '%f\n');
    errorGramMatrixSVD(1: cols(i), i) = abs(singularValuesHost - singularValuesGramMatrixSVD);
    
    fileID = fopen("../SingularValues/CudaDevice/OneSidedSequential/Singular Values Cuda Device "+matrixName(i)+".txt", 'r');
    singularValuesOneSidedNoRR = fscanf (fileID, '%f\n');
    errorSequential(1: cols(i), i) = abs(singularValuesHost - singularValuesOneSidedNoRR);
    
    fileID = fopen("../SingularValues/CudaDevice/OneSidedSequentialCublas/Singular Values Cuda Device "+matrixName(i)+".txt", 'r');
    singularValuesOneSidedNoRRCublas = fscanf (fileID, '%f\n');
    errorSequentialCublas(1: cols(i), i) = abs(singularValuesHost - singularValuesOneSidedNoRRCublas);
    
    fileID = fopen("../SingularValues/CudaDevice/OneSidedParallelSemiShared/Singular Values Cuda Device "+matrixName(i)+".txt", 'r');
    singularValuesOneSidedRR = fscanf (fileID, '%f\n');
    errorParallelSemiShared(1: cols(i), i) = abs(singularValuesHost - singularValuesOneSidedRR);
    
    fileID = fopen("../SingularValues/CudaDevice/OneSidedParallelGlobal/Singular Values Cuda Device "+matrixName(i)+".txt", 'r');
    singularValuesOneSidedRRnoShared = fscanf (fileID, '%f\n');
    errorParallelGlobal(1: cols(i), i) = abs(singularValuesHost - singularValuesOneSidedRRnoShared);
    
    fileID = fopen("../SingularValues/CudaDevice/OneSidedParallelShared/Singular Values Cuda Device "+matrixName(i)+".txt", 'r');
    singularValuesOneSidedRRShared = fscanf (fileID, '%f\n');
    errorParallelShared(1: cols(i), i) = abs(singularValuesHost - singularValuesOneSidedRRShared);
    
    %% load time files
    fileID = fopen("../Time/CudaHost/Time "+matrixName(i)+".txt",'r');
    timeHost(i) = fscanf (fileID, '%f');
    
    fileID = fopen("../Time/CudaDevice/GramMatrixSVD/Time "+matrixName(i)+".txt",'r');
    timeDeviceGramMatrixSVD(i) = fscanf (fileID, '%f');
     
    fileID = fopen("../Time/CudaDevice/OneSidedSequential/Time "+matrixName(i)+".txt",'r');
    timeDeviceSequential(i) = fscanf (fileID, '%f');

    fileID = fopen("../Time/CudaDevice/OneSidedSequentialCublas/Time "+matrixName(i)+".txt",'r');
    timeDeviceSequentialCublas(i) = fscanf (fileID, '%f');
    
    fileID = fopen("../Time/CudaDevice/OneSidedParallelSemiShared/Time "+matrixName(i)+".txt",'r');
    timeDeviceParallelSemiShared(i) = fscanf (fileID, '%f');
    
    fileID = fopen("../Time/CudaDevice/OneSidedParallelGlobal/Time "+matrixName(i)+".txt",'r');
    timeDeviceParallelGlobal(i) = fscanf (fileID, '%f');
    
    fileID = fopen("../Time/CudaDevice/OneSidedParallelShared/Time "+matrixName(i)+".txt",'r');
    timeDeviceParallelShared(i) = fscanf (fileID, '%f');
    
end

%% Mean square error plot
figure('Name','Mean Square Error','NumberTitle','off');
for i = 1 : length(matrixName)
    %scatter (cols (i), sum(errorMatlab (:, i).^2)/cols (i), 'b-+');
    %hold on;
    %scatter (cols (i), sum(errorGramMatrixSVD (:, i).^2)/cols (i), 'r*');
    %hold on;
    scatter (cols (i), sum(errorSequential (:, i).^2)/cols (i), 'go');
    hold on;
    %scatter (cols (i), sum(errorSequentialCublas (:, i).^2)/cols (i), 'ksquare');
    %hold on;
    scatter (cols (i), sum(errorParallelSemiShared (:, i).^2)/cols (i), 'rx');
    hold on;
    scatter (cols (i), sum(errorParallelGlobal (:, i).^2)/cols (i), 'm>');
    hold on;
    scatter (cols (i), sum(errorParallelShared (:, i).^2)/cols (i), 'k<');
    hold on;    
end
%legend('Gram Matrix SVD', 'One Sided No RR', 'errorOneSidedNoRRCublas', 'One Sided RR', 'One Sided RR no Shared', 'One Sided RR Shared', 'Location', 'northwest'); 
legend('Sequential', 'Parallel Semi Shared', 'Parallel Global', 'Parallel Shared', 'Location', 'northwest'); 
xlabel('Columns');
ylabel('Mean Square Error');

%% Time plot
figure('Name','Columns vs Time','NumberTitle','off');
%plot (matrixDim (:, 2), timeMatlabOneSided);
%hold on;
%plot (matrixDim (:, 2), timeMatlabSVD);
%hold on;
plot (matrixDim (:, 2), timeHost, 'b-+');
hold on;
%plot (matrixDim (:, 2), timeDeviceGramMatrixSVD, 'r-*');
%hold on;
plot (matrixDim (:, 2), timeDeviceSequential, 'g-o');
hold on;
%plot (matrixDim (:, 2), timeDeviceSequentialCublas, 'k-.square');
%hold on;
plot (matrixDim (:, 2), timeDeviceParallelSemiShared, 'r-x');
hold on;
plot (matrixDim (:, 2), timeDeviceParallelGlobal, 'm--');
hold on;
plot (matrixDim (:, 2), timeDeviceParallelShared, 'k:');

legend('Host', 'Sequential', 'Parallel Semi Shared', 'Parallel Global', 'Parallel Shared', 'Location', 'northwest'); 
xlabel('Columns');
ylabel({'Time','(in millisecond)'})

%% Setup the Import Options and import the data from One Sided Sequential
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = [7, 10; 15, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["TimePerc", "Time", "Calls", "Avg", "Min", "Max", "Name"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Name", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Name", "EmptyFieldRule", "auto");

%% Profiling Plot One Sided Sequential
for i = 1 : length(matrixName)
    % Import the data
    Profiling = readtable("../Profiling/OneSidedSequential/Prof"+matrixName(i)+".csv", opts);
    kernelPerc(:, i) = table2array(Profiling(1:4,1));
    kernelName(:, i) = table2array(Profiling(1:4,7));
    APIPerc(:, i) = table2array(Profiling(5:end,1));
    APIName(:, i) = table2array(Profiling(5:end,7));
end

%reordering kernel matrix
for c = 2 : length(matrixName)
    for i = 1 : length(kernelName(:, c))
        for j = i+1 : length(kernelName(:, c))
            if (kernelName(i, c) ~= kernelName(i, 1))
                tempName = kernelName(i, c);      
                kernelName(i, c) = kernelName(j, c);
                kernelName(j, c) = tempName;

                tempPerc = kernelPerc(i, c);
                kernelPerc(i, c) = kernelPerc(j, c);
                kernelPerc(j, c) = tempPerc;
            end
        end
    end
end

%reordering API matrix
for c = 2 : length(matrixName)
    for i = 1 : length(APIName(:, c))
        for j = i+1 : length(APIName(:, c))
            if (APIName(i, c) ~= APIName(i, 1))
                tempName = APIName(i, c);      
                APIName(i, c) = APIName(j, c);
                APIName(j, c) = tempName;

                tempPerc = APIPerc(i, c);
                APIPerc(i, c) = APIPerc(j, c);
                APIPerc(j, c) = tempPerc;
            end
        end
    end
end

figure('Name','API Profiling Sequential','NumberTitle','off');
bar(cols, APIPerc,'stacked');
xlabel('Columns');
ylabel({'Time (%)'});
legend(APIName(:,1), 'Location', 'bestoutside');

figure('Name','Kernel Profiling Sequential','NumberTitle','off');
bar(cols, kernelPerc,'stacked');
xlabel('Columns');
ylabel({'Time (%)'});
legend(kernelName(:,1), 'Location', 'bestoutside');

%Clear temporary variables
clear opts

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = [7, 11; 16, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["TimePerc", "Time", "Calls", "Avg", "Min", "Max", "Name"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Name", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Name", "EmptyFieldRule", "auto");

%% Profiling Plot One Sided Parallel Global
clear kernelPerc;
clear kernelName;
clear APIPerc;
clear APIName;
clear Profiling;

for i = 1 : length(matrixName)
    % Import the data
    Profiling = readtable("../Profiling/OneSidedParallelGlobal/Prof"+matrixName(i)+".csv", opts);
    kernelPerc(:, i) = table2array(Profiling(1:5,1));
    kernelName(:, i) = table2array(Profiling(1:5,7));
    APIPerc(:, i) = table2array(Profiling(6:end,1));
    APIName(:, i) = table2array(Profiling(6:end,7));
end

%reordering kernel matrix
for c = 2 : length(matrixName)
    for i = 1 : length(kernelName(:, c))
        for j = i+1 : length(kernelName(:, c))
            if (kernelName(i, c) ~= kernelName(i, 1))
                tempName = kernelName(i, c);      
                kernelName(i, c) = kernelName(j, c);
                kernelName(j, c) = tempName;

                tempPerc = kernelPerc(i, c);
                kernelPerc(i, c) = kernelPerc(j, c);
                kernelPerc(j, c) = tempPerc;
            end
        end
    end
end

%reordering API matrix
for c = 2 : length(matrixName)
    for i = 1 : length(APIName(:, c))
        for j = i+1 : length(APIName(:, c))
            if (APIName(i, c) ~= APIName(i, 1))
                tempName = APIName(i, c);      
                APIName(i, c) = APIName(j, c);
                APIName(j, c) = tempName;

                tempPerc = APIPerc(i, c);
                APIPerc(i, c) = APIPerc(j, c);
                APIPerc(j, c) = tempPerc;
            end
        end
    end
end

figure('Name','API Profiling Parallel Global','NumberTitle','off');
bar(cols, APIPerc,'stacked');
xlabel('Columns');
ylabel({'Time (%)'});
legend(APIName(:,1), 'Location', 'bestoutside');

figure('Name','Kernel Profiling Parallel Global','NumberTitle','off');
bar(cols, kernelPerc,'stacked');
xlabel('Columns');
ylabel({'Time (%)'});
legend(kernelName(:,1), 'Location', 'bestoutside');

%% Profiling Plot One Sided Parallel Semi Shared
clear kernelPerc;
clear kernelName;
clear APIPerc;
clear APIName;
clear Profiling;

for i = 1 : length(matrixName)
    % Import the data
    Profiling = readtable("../Profiling/OneSidedParallelSemiShared/Prof"+matrixName(i)+".csv", opts);
    kernelPerc(:, i) = table2array(Profiling(1:5,1));
    kernelName(:, i) = table2array(Profiling(1:5,7));
    APIPerc(:, i) = table2array(Profiling(6:end,1));
    APIName(:, i) = table2array(Profiling(6:end,7));
end

%reordering kernel matrix
for c = 2 : length(matrixName)
    for i = 1 : length(kernelName(:, c))
        for j = i+1 : length(kernelName(:, c))
            if (kernelName(i, c) ~= kernelName(i, 1))
                tempName = kernelName(i, c);      
                kernelName(i, c) = kernelName(j, c);
                kernelName(j, c) = tempName;

                tempPerc = kernelPerc(i, c);
                kernelPerc(i, c) = kernelPerc(j, c);
                kernelPerc(j, c) = tempPerc;
            end
        end
    end
end

%reordering API matrix
for c = 2 : length(matrixName)
    for i = 1 : length(APIName(:, c))
        for j = i+1 : length(APIName(:, c))
            if (APIName(i, c) ~= APIName(i, 1))
                tempName = APIName(i, c);      
                APIName(i, c) = APIName(j, c);
                APIName(j, c) = tempName;

                tempPerc = APIPerc(i, c);
                APIPerc(i, c) = APIPerc(j, c);
                APIPerc(j, c) = tempPerc;
            end
        end
    end
end

figure('Name','API Profiling Parallel Semi Shared','NumberTitle','off');
bar(cols, APIPerc,'stacked');
xlabel('Columns');
ylabel({'Time (%)'});
legend(APIName(:,1), 'Location', 'bestoutside');

figure('Name','Kernel Profiling Parallel Semi Shared','NumberTitle','off');
bar(cols, kernelPerc,'stacked');
xlabel('Columns');
ylabel({'Time (%)'});
legend(kernelName(:,1), 'Location', 'bestoutside');


%% Profiling Plot One Sided Parallel Shared
clear kernelPerc;
clear kernelName;
clear APIPerc;
clear APIName;
clear Profiling;

for i = 1 : length(matrixName)
    % Import the data
    Profiling = readtable("../Profiling/OneSidedParallelShared/Prof"+matrixName(i)+".csv", opts);
    kernelPerc(:, i) = table2array(Profiling(1:5,1));
    kernelName(:, i) = table2array(Profiling(1:5,7));
    APIPerc(:, i) = table2array(Profiling(6:end,1));
    APIName(:, i) = table2array(Profiling(6:end,7));
end

%reordering kernel matrix
for c = 2 : length(matrixName)
    for i = 1 : length(kernelName(:, c))
        for j = i+1 : length(kernelName(:, c))
            if (kernelName(i, c) ~= kernelName(i, 1))
                tempName = kernelName(i, c);      
                kernelName(i, c) = kernelName(j, c);
                kernelName(j, c) = tempName;

                tempPerc = kernelPerc(i, c);
                kernelPerc(i, c) = kernelPerc(j, c);
                kernelPerc(j, c) = tempPerc;
            end
        end
    end
end

%reordering API matrix
for c = 2 : length(matrixName)
    for i = 1 : length(APIName(:, c))
        for j = i+1 : length(APIName(:, c))
            if (APIName(i, c) ~= APIName(i, 1))
                tempName = APIName(i, c);      
                APIName(i, c) = APIName(j, c);
                APIName(j, c) = tempName;

                tempPerc = APIPerc(i, c);
                APIPerc(i, c) = APIPerc(j, c);
                APIPerc(j, c) = tempPerc;
            end
        end
    end
end

figure('Name','API Profiling Parallel Shared','NumberTitle','off');
bar(cols, APIPerc,'stacked');
xlabel('Columns');
ylabel({'Time (%)'});
legend(APIName(:,1), 'Location', 'bestoutside');

figure('Name','Kernel Profiling Parallel Shared','NumberTitle','off');
bar(cols, kernelPerc,'stacked');
xlabel('Columns');
ylabel({'Time (%)'});
legend(kernelName(:,1), 'Location', 'bestoutside');

%Clear temporary variables
clear opts
