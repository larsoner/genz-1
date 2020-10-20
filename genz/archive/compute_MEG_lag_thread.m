sfreq = 300;


% Auto-correlation
[acc1,cclags] = xcorr(wernicke,wernicke,'normalized');  % Wernicke
[pxx1,ff] = pwelch(wernicke,sfreq);  % PSD

[acc2,cclags] = xcorr(broca,broca,'normalized');  % Broca
[pxx2,ff] = pwelch(broca,sfreq); 
tvec = linspace(0,5,1001)';  % time minutes
cctime = (cclags / sfreq);  % time vector (sec)

figure(1)
tl = tiledlayout(2,2);
ax1 = nexttile;
plot(ax1, cctime, acc1, 'r')
ax2 = nexttile;
plot(ax2, cctime, acc2, 'b')
ax3 = nexttile;
plot(ax3, ff, pxx1, 'r')
ax4 = nexttile;
plot(ax4, ff, pxx2, 'b')


% cross-correlation
[cc, cclags] = xcorr(wernicke, broca, 'normalized'); 
cctime = (cclags / sfreq);  % time vector (sec)

[pxx_cross,ff] = cpsd(wernicke,broca,sfreq);
[peak_lag, peak_cov] = parabolic_interp(cc,sfreq);
compute_lag_thread

figure(2)
tl = tiledlayout(3,1);
% signals
ax1 = nexttile;
plot(ax1, tvec', broca, 'b', tvec, wernicke, 'r')
% correllalogram
ax2 = nexttile;
stem(ax2, cctime', cc);
%hold on
%[ d, ix ] = min( abs( cctime-peak_lag ) );
%scatter(ax2,ix, peak_cov)
%hold off
ax3 = nexttile;
plot(ax3, ff, abs(pxx_cross));
