This is my working code for looking at velocity data from lab gravity currents
obtained through PIV. At some point this might be stable but right now it is a
work in progress!

Things to do:

- [ ] Non-dimensionalise

- [ ] vorticity plots. basically port demo/plot.py into the plotter
  class

- [ ] can we identify qualitatively different regions of the flow?
    - [ ] do they have distinct pdfs?
    - [ ] does cantero have something to say here?

- [ ] DMD: can we recompose the flow from low order modes? are the
  stats the same?


More things to do:

- [ ] look at ogive plots

- [ ] distinguish between sampling dimension and time / space in
  front relative extraction.

- [ ] wavelet ensembles? can we increase confidence with more
  ensemble members

- [ ] make the pdf as a function of time and height work
    - [ ] plot with log height
    - [ ] plot for multiple ensembles

- [ ] distinguish ensembles - inter / intra run, inter parameter

- [ ] rapid distortion theory. eddy turnover time large compared
  with advective timescale?

- [ ] pdfs limited to particular events (eddies)

- [ ] fit log profile to vertical pdfs (log height)

- [ ] compute vertical pdfs with highlighted data exceeding certain
  percentile close together in space / time (i.e. same event)


Done:

- [x] interpolate zeros in pre processor - how does pandas do it?

- [x] fit straight line / smooth front detection
    - [x] check sensitivity of stats to different fittings

- [x] overlay multiple runs to get first impression of similarity
  (use the single height over time)

- [x] do the front relative transform along the other (space) axis
  and see what it looks like

- [x] look at streamwise velocity
- [x] subtract front speed to get front relative velocity
- [x] is there a region of the flow in which the decomp mean front
  relative velocity is zero? (is this the region of statistical
  stationarity?)
