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
