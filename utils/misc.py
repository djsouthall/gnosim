'''
Misc functions that might be useful to reuse.
'''

import numpy
import pylab
import os
import os.path
import glob


def getColorMap(n):
	'''
	Returns a list of unique colors which can be used when plotting.  Does not include white.

	Parameters
	----------
	n : int
		The number of colors to include in the color map.

	Returns
	-------
		colors : list of tuples
			Each tuple represents a color for plotting.
	'''
	colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
	colors = [colormap(i) for i in numpy.linspace(0, 1,n+1)]
	return colors

def makeIndexHTML(path = './',filetype = 'svg'):
    '''
    Makes a crude html image browser of the created images to be loaded in a web browser. Image filytpe should not have the . and Path should have / at the end

    Parameters
    ----------
    path : str, optional
        The path to the folder containing all of the images to be indexed.  Should not contain a forward slash at the end.
    filetye : str, optional
        The file type extension of the images to be indexed.  Should not contain the '.'. 


    '''
    header = os.path.realpath(path).split('/')[-1]
    infiles = glob.glob('%s*%s'%(path,filetype))
    
    infiles_num = []
    
    for infile in infiles:
        if len(infile.split('-event')) > 1:
            infiles_num.append(int(infile.split('-event')[-1].replace('.' + filetype,'')))
        else:
            infiles_num.append(-1) #will put all non-conforming files at front before sorted event files.
    infiles = numpy.array(infiles)[numpy.argsort(infiles_num)] #sorts files in index by event number
        
    #I want to sort by event number here!
    image_list = ''
    for infile in infiles:
        image_list = image_list + '\t<img class="mySlides" src="' + infile.split('/')[-1] + '" style="width:100%">\n'
    
    #print(image_list)
    
    
    template =  '''
                <!DOCTYPE html>
                <html>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
                <style>
                .mySlides {display:none;}
                </style>
                <body>
                
                <head>
                <title> RCC KICP | Dan Southall </title>
                </head>
                <p><strong> Dan Southall </strong> | <a href="https://kicp.uchicago.edu/people/profile/daniel_southall.html"> KICP Profile </a> | <a href="../../index.html"> Home </a></p>
                
                <h2 class="w3-center"><strong> """ + header + """</strong></h2>
                
                <input id="slide_index" size="4" value="1" onchange="showDivs(parseInt(document.getElementById('slide_index').value))">
                
                <div class="w3-content w3-display-container"> 
                """ + image_list + """
                </div>
                
                <button class="w3-button w3-black w3-display-left" onclick="plusDivs(-1)">&#10094;</button>
                <button class="w3-button w3-black w3-display-right" onclick="plusDivs(1)">&#10095;</button>
                
                </div>
                <script>
                var slideIndex = 1;
                showDivs(slideIndex);

                function plusDivs(n) {
                  showDivs(slideIndex += n);
                }

                function showDivs(n) {
                  var i;
                  var x = document.getElementsByClassName("mySlides");
                  slideIndex =n;
                  if (n > x.length) {slideIndex = 1}    
                  if (n < 1) {slideIndex = x.length}
                  for (i = 0; i < x.length; i++) {
                     x[i].style.display = "none";  
                  }
                  x[slideIndex-1].style.display = "block"; 
                  document.getElementById("slide_index").value = slideIndex;
                  location.hash = "#" + slideIndex;
                  document.getElementById("filename").innerHTML = x[slideIndex-1].getAttribute("src");
                }
                
                function load() 
                {
                  var maybe = parseInt(location.hash.slice(1));
                  if (!isNaN(maybe)) 
                  {
                    showDivs(maybe); 
                  }
                  else showDivs(1); 
                }
                </script>

                </body>
                </html>
                '''
    print(template)
    outfile_name = path + 'index'
    if os.path.isfile(outfile_name +'.html'):
        print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(outfile_name))
        outfile_name = outfile_name + '_new'
        while os.path.isfile(outfile_name+'.html'):
            outfile_name = outfile_name + '_new'
    outfile = open(outfile_name + '.html','w')
    outfile.write(template)
    outfile.close()