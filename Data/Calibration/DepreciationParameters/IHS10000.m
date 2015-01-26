function invert=IHS10000(x,theta,y_hat);

dist= (-y_hat).*theta+log(theta.*(x./10000)+sqrt(1+(theta.^2).*((x./10000).^2)));
invert=[dist];
