'''
CLASS FOR ANNOTATE EVENTS IN ONE DIMENSIONAL SIGNAL
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Annotation_Events:
   def __init__(self,file,sheet_name,column_name, output_file):
        df = pd.read_excel(file, sheet_name = sheet_name)
        self.signal = df[column_name].values
        self.output_file = output_file
        if os.path.exists(self.output_file):
            # load prior labels
            old = pd.read_excel(self.output_file + "labels.xlsx")
            self.labels = old['Labels'].values
            # load prior motif regions
            mot = pd.read_excel(self.output_file + "motifs.xlsx")
            self.motif_regions = list(zip(mot.start, mot.end, mot.label))
        else:
            self.labels = np.zeros_like(self.signal, dtype=int)
            self.motif_regions = []
            
        self.timestamp = df.index.values
        self.clicks = []
        self.vlines = []
        self.texts = []
        self.spans = []
        self._init_plot()
        self._draw_existing_regions()
    
    #=== Plot the graph ====
   def _init_plot(self):
       self.fig, self.ax = plt.subplots(figsize = (15,4))
       self.ax.plot(self.signal, label = "Signal", color = "Blue")
       self.ax.set_title("Click twice to mark region. Press number key to label. 'd' to undo. 'x' to remove last line")      
       self.ax.set_xlabel("Frame")
       self.ax.set_ylabel("Signal")
       self.ax.grid(True)
       self.fig.canvas.mpl_connect('button_press_event', self.onclick)
       self.fig.canvas.mpl_connect('key_press_event', self.onkey)
    
       
    #==== when press button clicks are accumulated======
   def onclick(self, event):
       if event.inaxes != self.ax or event.button != 1: #only left button of the mouse can use   
           return
       x= int(event.xdata)
       self.clicks.append(x)
       v = self.ax.axvline(x=x, color ='red', linestyle = '--')
       self.vlines.append(v)
       self.fig.canvas.draw() #update plot
       
    #=====label when press a digit=======
   def onkey(self, event):
        if event.key.isdigit() and len(self.clicks) >= 2:
            self._label_last_region(int(event.key))
        elif event.key == 'd':
            self._delete_last_region()
        elif event.key == 'x':
            self._delete_last_line()
     
                  
    #===label last region and print===
   def _label_last_region(self, label):
        cmap1 = ['white','red', 'green','blue','magenta','cyan','yellow','brown']
        start, end = sorted(self.clicks[-2:])
        self.labels[start:end] = label
        self.motif_regions.append((start, end, label))
        span = self.ax.axvspan(start, end, color = cmap1[label], alpha=0.3)
        self.spans.append(span)
        
        print(f"Labeled region {start}-{end} with {label}")
        t = self.ax.text((start + end) // 2, np.max(self.signal)*0.95, f"{label}", color = 'red', ha = 'center')
        self.texts.append(t)
        self.clicks.clear()
        self.fig.canvas.draw() #update
        
    #====plot saved labels ==================
   def _draw_existing_regions(self):
        """Paint old spans, lines, and labels onto the axes."""
        cmap1 = ['white','red','green','blue','magenta','cyan','yellow','brown']
        ymax = np.max(self.signal)*0.95
        for start, end, label in self.motif_regions:
            # vertical lines
            # v1 = self.ax.axvline(start, color='white', linestyle='--')
            # v2 = self.ax.axvline(end,   color='white', linestyle='--')
            # self.vlines += [v1, v2]
            # shaded span
            span = self.ax.axvspan(start, end, color=cmap1[label], alpha=0.3)
            self.spans.append(span)
            # text label
            t = self.ax.text((start+end)//2, ymax, str(label),
                             color='red', ha='center')
            self.texts.append(t)
        self.fig.canvas.draw()

    #==================Remove lines ==========================#
   def _delete_last_line(self):
        # only respond to picking one of our vlines
        self.vlines.pop().remove()
        self.clicks.pop()
        self.fig.canvas.draw() #update
        
    #======delete last region ===========#
   def _delete_last_region(self):
        if not self.motif_regions:
            print("Nothing to delete")
            return
        start, end, label = self.motif_regions.pop() #get last elements of the list
        self.labels[start:end] = 0
        print(f"Deleted region {start}-{end} (label {label})")
        #remove the last 2 lines
        for _ in range(2):
            if self.vlines:
                self.vlines.pop().remove()
            
            if self.texts:
                self.texts.pop().remove()
                
            if self.spans:
                self.spans.pop().remove()
              
        
        self.fig.canvas.draw()
    
    #==================save the labels======================#      
   def _save_outputs(self):
       data = pd.DataFrame({"Frame": self.timestamp, "Signal": self.signal, "Labels": self.labels})
       with pd.ExcelWriter(self.output_file + "labels.xlsx", engine='openpyxl',mode='w') as writer:
         data.to_excel(writer, index = False)
       
    #===================save motifs regions ==================================
   def _save_motifs(self):
     data = pd.DataFrame(self.motif_regions, columns = ['start', 'end', 'label'])
     with pd.ExcelWriter(self.output_file + "motifs.xlsx", engine='openpyxl',mode='w') as writer:
       data.to_excel(writer, index = False)
          
   def __call__(self):
      plt.tight_layout()
      plt.show()
      self._save_outputs()
      self._save_motifs()
   
      
      
       