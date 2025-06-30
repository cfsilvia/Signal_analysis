'''
CLASS FOR ANNOTATE EVENTS IN ONE DIMENSIONAL SIGNAL
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Annotation_Events:
   def __init__(self,file,sheet_name,column_name, output_file):
        df = pd.read_excel(file, sheet_name = sheet_name)
        self.signal = df[column_name].values
        self.labels = np.zeros_like(self.signal, dtype = int)
        self.timestamp = df.index.values
        self.output_file = output_file
        self.motif_regions = []
        self.clicks = []
        self.vlines = []
        self.texts = []
        self._init_plot()
    
    #=== Plot the graph ====
   def _init_plot(self):
       self.fig, self.ax = plt.subplots(figsize = (15,4))
       self.ax.plot(self.signal, label = "Signal", color = "Blue")
       self.ax.set_title("Click twice to mark region. Press number key to label. 'd' to undo. 'x' to remove last line")      
       self.ax.set_xlabel("Frame")
       self.ax.set_ylabel("Amplitude")
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
        start, end = sorted(self.clicks[-2:])
        self.labels[start:end] = label
        self.motif_regions.append((start, end, label))
        print(f"Labeled region {start}-{end} with {label}")
        t = self.ax.text((start + end) // 2, np.max(self.signal)*0.95, f"{label}", color = 'red', ha = 'center')
        self.texts.append(t)
        self.clicks.clear()
        self.fig.canvas.draw() #update

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
        self.fig.canvas.draw()
    
    #==================save the labels======================#      
   def _save_outputs(self):
       data = pd.DataFrame({"Frame": self.timestamp, "Signal": self.signal, "Labels": self.labels})
       data.to_excel(self.output_file + "labels.xlsx", index = False)
       
    #===================save motifs regions ==================================
   def _save_motifs(self):
     data = pd.DataFrame(self.motif_regions, columns = ['start', 'end', 'label'])
     data.to_excel(self.output_file + "motifs.xlsx", index = False)
          
   def __call__(self):
      plt.tight_layout()
      plt.show()
      self._save_outputs()
      self._save_motifs()
   
      
      
       