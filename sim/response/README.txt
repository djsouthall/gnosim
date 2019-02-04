I was sent these files from Dan Smith, who was working on pulsed data for ARA. 
He obtained this info from that data.  Here is his description from the email:

"
Attached is the current antenna / electronic chain response. I saved them in 
numpy data format (npy) and included a quick script on how to load and it and 
convert it to time domain.  Important to note that frequencies above ~750 and 
below ~150 MHz are aphysical (system couldn't actually read them) and the values 
that are in those ranges exist just to make the signal causal. I am not sure how 
you should deal with that in your simulation, just wanted to let you know.
"

The v2 files came with the message:

"
Attached are the renormalized functions. The antenna impulse response shouldn't 
have changed much, but the electronics chain should be Much larger, 80 dB larger. 
I am still trying to iron out various factors of 2 and Pi, but those are ~5% 
errors in dB, so the attached files should be close to (or are) the final product.  
"

Dan said in person that he still has to double check with Cosmin about the specifics
of his calculation, final factors, and if he is in SI or gaussian units for the
calculation.  

v3 and v4 are functionally the same but with different binning.  Dan Smith is
still trying to get my a finalized version.  I will be making a tool to scale
the system response such that the noise matches the levels seen in the
experiement, but for the antenna response I still need Dan to get it for me.

v5 has scaled system response to match expected levels of noise. 

v6 Comes from Dan Smith with the Update message:
"
... One difference here (besides the, hopefully, correct scaling) is that I am giving 
you the response in the number of points that the digitizer reads out 
(512 time domain, 257 freq domain). This is just to maintain consistency on my end: 
its easiest for me if everything stays in the unpadded state as to not accidentally 
change the amount of power in a signal.
"

v6 appeared to produce acausal signals, Dan Smith indicated this was likely a
result of how he was sampling.  This has been fixed in v7.  Message from Dan
Smith describing the problem:
"What went wrong[in v6]: I had a time-domain zero padding and I wanted to get back to
the original length but instead of doing the smart thing of truncating the
zeros in the time domain, I picked pts in the freq domain. I did that because
I was accidentally cutting signal when removing zeros in the time domain
making the signal in freq domain wonky, so I thought I could out smart it. I
was wrong because my picking pts in the freq domain to down sample was
acausal, so I just was more careful removing zeros in the time domain and hey
presto works out."
