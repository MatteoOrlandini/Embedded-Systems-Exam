function singvals = One_Sided_Jacobi_Rotation(A)

myeps = 0.0001;
%returns the lengths of the specified dimensions in a row vector
M = size(A, 1); 
N = size(A, 2);

singvals = zeros(N, 1);

tao = 0;
U = A;
conv_count = 100;
alpha = 0;
beta = 0;
gamma = 0;

%printf("\n\nOriginal A\n", i-1, j-1);
%A
exit_flag = false;
while ~exit_flag && conv_count > 0
  exit_flag = true;
  %printf("\n\n----------------------------------   %d   ----------------------------------", conv_count);
  for j = N:-1:2
    for i = j-1:-1:1
    %alpha: norma al quadrato di U(:,i)
    alpha = 0;
    %beta: norma al quadrato di U(:,j)
    beta = 0;
    %gamma: prodotto scalare di U'(:,i) e U(:,j)
    gamma = 0;
      for k = M:-1:1
        %con k scorro le righe dei vettori U(:,i) e U(:,j)
        alpha = alpha + U(k, i) * U(k, i);
        beta = beta + U(k, j) * U(k, j);
        gamma = gamma + U(k, i) * U(k, j);
      end
      %off(A)
      limit = abs(gamma) / sqrt(alpha*beta);

      if(limit > myeps)
        exit_flag = false;
      end

      tao = (beta - alpha) / (2 * gamma);
      t = sign(tao) / (abs(tao) + sqrt(1 + tao^2));
      c = 1 / ( sqrt(1 + t^2) );
      s = c*t;

      %printf("\n____________________________________________________________________________________");
      %printf("\n[alpha:%f, beta:%f, gamma:%f, tao:%f, limit:%f]\t\t|(i,j):(%d,%d)", alpha, beta, gamma, tao, limit, i-1, j-1);
      %printf("\n[t:%f, c:%f, s:%f]\n", t, c, s);

      t = U(:, i);
      U(:, i) = c*t - s*U(:, j);
      U(:, j) = s*t + c*U(:, j);

      for y=1:N
        %printf("\t%f", U(1:5, y));
        %printf("\t%f", U(6:M, y));
        %printf("\n");
      end
      %printf("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n\n\n");
    end
    %printf("\n(j):(%d)", j-1);
  end
  conv_count = conv_count - 1;
end


%printf("\n\n\n____________________________________________________________________________________\n");
%printf("____________________________________OUT OF DO_______________________________________\n");
%for y=1:N
  %printf("\t%f", U(1:5, y));
  %printf("\t%f", U(6:M, y));
  %printf("\n");
%end
%printf("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n\n\n");

for j=N:-1:1
  singvals(j) = sqrt(dot(U(:,j), U(:,j)));
end

u=U;

singvals = sort(singvals, 'descend');
%singvals'

%[U2, S2, V2] = svd(A);
%diag(S2)

if(exit_flag)
  %fprintf("\n-----------------------STOP AVVENUTO PER SUPERAMENTO LIMITE-----------------------\n");
  %fprintf("LIMIT (eps:%f):\t%2.30f", myeps, limit);
  %fprintf("\nN° CICLI COMPIUTI:\t%d", 100-conv_count);
  %fprintf("\nN° CICLI RIMASTI:\t%d\n", conv_count);
else
  %fprintf("\n-----------------------STOP AVVENUTO PER CONVCOUNT-----------------------\n");
endif
%endfunction

end
