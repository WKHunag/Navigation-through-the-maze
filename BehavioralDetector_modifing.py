import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json


class BehaviorDetector:
    def __init__(self, json_path):
        self.json_path = json_path
        if os.path.isfile(self.json_path):
            with open(self.json_path, "r") as f:
                jsondata = json.load(f)
                self.actpixel_path = jsondata['active pixel path']
                self.backimg_path = self.actpixel_path.replace('.txt','_backimg.png').replace('pixel data', 'background image')
                self.backImg = cv2.imread(self.backimg_path)
                self.imgshape = self.backImg.shape
                self.fps = jsondata['fps']
                self.time = jsondata['time']
                self.frames1sthr = len(pd.read_csv(self.actpixel_path, sep='\t', header=None, skiprows = 3600*2*self.fps*(self.time-1)))
                self.save_dir = jsondata['save_dir']
                self.kernel = np.array(jsondata['kernel'],np.uint8)
                self.minsize = jsondata['minsize']
                sftiming_path = self.actpixel_path.replace('.txt', '_timing.txt')
                if os.path.isfile(sftiming_path):
                    self.df_timing = pd.read_csv(sftiming_path, header=None)
                self.arm_nums = np.array(jsondata['# of arms'])
                self.arms_roi = np.array(jsondata['arms_roi'])
                self.omega_rois = jsondata['omega shape rois']
            f.close()
        else:
            print("The json file doesn't exist.")

    def Loaddata(self, actpixel_path, fps, time, save_dir, kernel, minsize):
        self.actpixel_path = actpixel_path
        self.backimg_path = self.actpixel_path.replace('.txt','_backimg.png').replace('pixel data', 'background image')
        self.backImg = cv2.imread(self.backimg_path)
        self.imgshape = self.backImg.shape
        self.fps = fps
        self.time = time
        self.frames1sthr = len(pd.read_csv(self.actpixel_path, sep='\t', header=None, skiprows = 3600*2*self.fps*(self.time-1)))
        self.save_dir = save_dir
        self.kernel = kernel
        self.minsize = minsize
        sftiming_path = self.actpixel_path.replace('.txt', '_timing.txt')
        if os.path.isfile(sftiming_path):
            self.df_timing = pd.read_csv(sftiming_path, header=None)

    def SetRois(self):
        cv2.namedWindow('select arms roi', 0)
        arms_roi = cv2.selectROIs('select arms roi', cv2.imread(self.backimg_path))
        cv2.namedWindow('select omega rois', 0)
        omega_rois = cv2.selectROIs('select omega rois', cv2.imread(self.backimg_path))
        cv2.destroyAllWindows() 
        self.arms_roi = arms_roi
        self.omega_rois = omega_rois
        self.arm_nums = len(arms_roi)

    def CreateJson(self):
        with open(self.json_path, 'w') as f:
            dic = {
                'active pixel path':self.actpixel_path,
                'fps':self.fps,
                'time':self.time,
                'save_dir':self.save_dir,
                'kernel':self.kernel.tolist(),
                'minsize':self.minsize,
                'arms_roi':self.arms_roi.tolist(),
                'omega shape rois':self.omega_rois.tolist(),
                '# of arms':self.arm_nums
            }
            json.dump(dic, f, indent=4)
        f.close()

# Calculate the ratios of fish in the different tunnels respectively. And plot the ratios.
    def RatiosDetect(self):
        """
        Calculate the ratios of fish in the different tunnels respectively.
        """       
        ratios = []
        arms = len(self.arms_roi)
        min_size = self.minsize

        for a in range(1,1+arms):
            locals()['ratios'+str(a)] = []  

        for n in range(self.time//3):
            if n == 0:
                df = pd.read_csv(self.actpixel_path, sep='\t', nrows=self.frames1sthr+3600*2*2*self.fps, header=None)
            else:
                df = pd.read_csv(self.actpixel_path, sep='\t', nrows=3600*self.fps*2*3, skiprows=self.frames1sthr+3600*2*self.fps*(3*(n-1)+2), header=None)

            for m in range(len(df)//2//600): 
                for a in range(1,1+arms):
                    locals()['number'+str(a)] = 0
                if m == len(df)//2//600 - 1:
                    for j in range(600*m,len(df)//2):
                        armswfish = []
                        empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                        empty[df.iloc[j*2]] += 1
                        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(empty.reshape(self.imgshape[0],self.imgshape[1]), connectivity=8)
                        sizes = stats[1:, -1]; nb_components = nb_components - 1
                        img2 = np.zeros((output.shape))
                        for i in range(0, nb_components):
                            if sizes[i] >= min_size:
                                img2[output == i + 1] = 1
                        for a in range(1,1+arms):
                            if img2[ self.arms_roi[a-1][1]:self.arms_roi[a-1][1]+self.arms_roi[a-1][3], self.arms_roi[a-1][0]:self.arms_roi[a-1][0]+self.arms_roi[a-1][2]].sum() > 200:
                                armswfish.append(1)
                                locals()['number'+str(a)] += 1
                            else: armswfish.append(0)
                        if len(np.where(np.array(armswfish)==1)[0]) == 1:
                            locals()['number'+str(np.where(np.array(armswfish)==1)[0][0]+1)] += 1
                    for a in range(1,1+arms):
                        locals()['ratios'+str(a)].append(locals()['number'+str(a)]/(len(df)//2-600*m)*100)             
                else:
                    for j in range(600*m,600*(m+1)):
                        armswfish = []
                        empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                        empty[df.iloc[j*2]] += 1
                        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(empty.reshape(self.imgshape[0],self.imgshape[1]), connectivity=8)
                        sizes = stats[1:, -1]; nb_components = nb_components - 1
                        img2 = np.zeros((output.shape))
                        for i in range(0, nb_components):
                            if sizes[i] >= min_size:
                                img2[output == i + 1] = 1
                        for a in range(1,1+arms):
                            if img2[ self.arms_roi[a-1][1]:self.arms_roi[a-1][1]+self.arms_roi[a-1][3], self.arms_roi[a-1][0]:self.arms_roi[a-1][0]+self.arms_roi[a-1][2]].sum() > 200:
                                armswfish.append(1)
                                locals()['number'+str(a)] += 1     
                            else: armswfish.append(0)  
                        if len(np.where(np.array(armswfish)==1)[0]) == 1:
                            locals()['number'+str(np.where(np.array(armswfish)==1)[0][0]+1)] += 1   
                    for a in range(1,1+arms):
                        locals()['ratios'+str(a)].append(locals()['number'+str(a)]/600*100)
                    
        for a in range(1,1+arms):
            ratios.append(locals()['ratios'+str(a)])
        ratios = pd.DataFrame(np.array(ratios))
        index = [ 'region '+str(a) for a in range(1, arms+1)]
        ratios.index = index
        
        return ratios

    def PreferPlot(self, ratios, figsize, colors, fontsize, sftimes = [], save = False, name=None, dpi=150):
        """

        Parameters
        ----------------
        ratios : DataFrame

        figsize : tuple

        colors : list

        name : str

        times : list
            The time point of smart film is turning on. 

        save : boolean

        Return
        ----------------
        plot image
        """

        labels = [ 'region '+str(i) for i in range(1, len(ratios)+1)]
        fig, ax = plt.subplots(figsize=figsize)
        ax.stackplot(np.arange(1,len(ratios.iloc[0])+1), ratios, colors=colors, labels=labels)

        ax.legend(loc='upper left',ncol = 6, fontsize=fontsize+5)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        ax.set_ylabel('Ratio (%)', fontsize=fontsize+5)
        ax.set_xlabel('Time (hr)', fontsize=fontsize+5)
        ax.set_xlim(0,len(ratios.iloc[0]))
        ax.set_xticks(range(0,len(ratios.iloc[0]),120))
        ax.set_xticklabels(range(0,len(ratios.iloc[0])//120))
        ax.set_ylim(0,250)
        ax.bar(sftimes.values.reshape(-1)/600, 400, alpha=1, color='k', width=5)
        if save:
            plt.savefig(os.path.join(self.save_dir,name+'.png'), dpi=dpi)
        plt.show()


# Detect the order of fish entering the different tunnels.
    def SeqDetect(self, active_pixel: pd.DataFrame):
        arms = len(self.arms_roi)
        sequences = []
        framenum = []
        for m in range(len(active_pixel)//2): 
                empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                empty[active_pixel.iloc[m*2]] += 255
                nb_components, output, stats, _ = cv2.connectedComponentsWithStats(empty.reshape(self.imgshape[0],self.imgshape[1]), connectivity=8)
                sizes = stats[1:, -1]; nb_components = nb_components - 1
                img2 = np.zeros((output.shape))
                i = np.argmax(sizes)
                img2[output == i + 1] = 1
                for a in range(arms):
                    if img2[ self.arms_roi[a][1]:self.arms_roi[a][1]+self.arms_roi[a][3], self.arms_roi[a][0]:self.arms_roi[a][0]+self.arms_roi[a][2]].sum() > 200:
                        sequences.append( a + 1 )
                        framenum.append(m)
                    else: pass
        return sequences, framenum

    def SeqInDiffpart(self):
        """
        Detect the order of fish entering the different tunnels.
        """
        with open(self.json_path,'r') as f:
            jsondata = json.load(f)
        try:
            self.seqs = np.array(jsondata['seqs'])
            self.framenums = np.array(jsondata['framenums'])
            print('Already have.')
        except:
            seqs = []
            framenums = []
            for i in range(self.time):
                if i == 0:
                    active_pixel = pd.read_csv(self.actpixel_path, sep='\t', header=None, nrows=self.frames1sthr)
                else:
                    active_pixel = pd.read_csv(self.actpixel_path, sep='\t', header=None, nrows=3600*2*self.fps, skiprows=3600*2*self.fps*(i-1)+self.frames1sthr)
                seq, framenum = self.SeqDetect(active_pixel)
                seqs = np.concatenate([seqs, np.array(seq)])
                if i == 0:
                    framenums = np.concatenate([framenums, np.array(framenum)])
                else:
                    framenums = np.concatenate([framenums, np.array(framenum)+self.frames1sthr//2+3600*self.fps*(i-1)])
            self.seqs = seqs
            self.framenums = framenums
            jsondata.update({'seqs':seqs.tolist(),'framenums':framenums.tolist()})
            f.close()
            with open(self.json_path, 'w') as f:
                json.dump(jsondata, f, indent=4)
                f.close()

# Detect the probabilities of fish traveling from one tunnel to others respectively.
    def ProbaDetect(self, start_arm:int):
        """
        Detect the probabilities of fish traveling from one tunnel to others respectively.
        """
        start_idx = np.where(self.seqs == start_arm)[0]
        start_index = []
        for i in range(len(start_idx)-1):
            if start_idx[i+1] - start_idx[i] > 1:
                start_index.append(start_idx[i])
            elif start_idx[i+1] - start_idx[i] == 1 and self.framenums[start_idx[i+1]] - self.framenums[start_idx[i]] > 20:
                start_index.append(start_idx[i])
        next_arms = [self.seqs[i+1]
                            for i in start_index]
        starting_arms, starting_nums = np.unique(next_arms, return_counts=True)
        for i in range(1,self.arm_nums+1):
            if i not in starting_arms:
                starting_arms = np.append(starting_arms, i)
                starting_nums = np.append(starting_nums, 0)
        swimming_seqences = np.asarray((starting_arms, starting_nums), dtype=np.float64).T.copy()
        swimming_seqences = np.sort(swimming_seqences.view(dtype=[('index', np.float64), ('count', np.float64)]),order='index',axis=0)

        return swimming_seqences


    def ProbToDifArms(self):
        """
        Combine all probabilities that fish traveling from every tunnel.
        """
        ProbToDifArms=[]
        for i in range(1,self.arm_nums+1):
            ProbToDifArms.append(self.ProbaDetect(i))
        return ProbToDifArms

    def PlotProbToDifArms(self, ProbToDifArms, filename, colors=None,weights=None, fontsize=35, legendcols=2, dpi=150):
        with open(self.json_path, "r") as f:
            jsondata=json.load(f)
            try:
                weights = jsondata['weights']
                colors = jsondata['colors']
            except:
                if colors == None or weights == None:
                    print('The arguments colors and weights are None.')
                else:
                    colors = colors
                    weights = weights
                    jsondata.update({'colors':colors, 'weights':weights})
                    f.close()
                    with open(self.json_path, "w") as f:
                        json.dump(jsondata, f, indent=4)     
                        f.close()
            fig, ax = plt.subplots(8,1,figsize=(8,7*self.arm_nums), sharey=True, sharex=True)
            plt.yticks(range(0,201,50))
            plt.ylim(0,200)
            plt.xticks(range(1,9))
            plt.xlabel('Index of arms', fontsize=fontsize+10)
            ax[4].set_ylabel('Weighted Ratio', fontsize=fontsize+10)
            plt.rcParams['xtick.labelsize']=fontsize
            plt.rcParams['ytick.labelsize']=fontsize
            for i in range(self.arm_nums):
                ax[i].bar(ProbToDifArms[i]['index'].flatten(),ProbToDifArms[i]['count'].flatten()/ProbToDifArms[i]['count'].sum()*100*weights[i], color=colors[i], edgecolor='k', label='From arm '+str(i+1))
            plt.style.use('ggplot')
            fig.legend( ncol=legendcols, fontsize=fontsize-15, shadow=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir,filename), dpi=dpi)


# Detect fish's moving trace by maximum projection.
    def ActPixelCount(self, iter=1, time_min=None):
        counts=[]
        if 'RBin' in self.actpixel_path:
            for n in range(int(60//time_min)):
                dfs = pd.read_csv(self.actpixel_path, sep='\t', nrows=int(time_min*60*20*2), skiprows= int(time_min*60*20*2*n), header=None)
                locals()['heatmap'+str(n)] = np.zeros((self.imgshape[:2]))
                for i in range(int(len(dfs)//2)):
                        empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                        empty[dfs.iloc[i*2]] += 255
                        erosion = cv2.erode(empty.reshape(self.imgshape[0],self.imgshape[1]), self.kernel, iterations=1)
                        locals()['heatmap'+str(n)][erosion==255] += 1
                counts.append(locals()['heatmap'+str(n)])
        else:
            for n in range(self.time):
                if n == 0:
                        dfs = pd.read_csv(self.actpixel_path, sep='\t', nrows=self.frames1sthr, header=None)
                else:
                        dfs = pd.read_csv(self.actpixel_path, sep='\t', nrows=36000*2, skiprows=self.frames1sthr+36000*2*(n-1), header=None)
                locals()['heatmap'+str(n)] = np.zeros((self.imgshape[:2]))
                for i in range(int(len(dfs)//2)):
                        empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                        empty[dfs.iloc[i*2]] += 255
                        erosion = cv2.erode(empty.reshape(self.imgshape[0],self.imgshape[1]), self.imgshape, iterations=iter)
                        locals()['heatmap'+str(n)][erosion==255] += 1
                counts.append(locals()['heatmap'+str(n)])
        return counts

    def MovingTrace(self, activepixel_counts, figsize, cbarsize, fontsize, vmax=150, save=False, filename=None, dpi=150):
        for n in range(len(activepixel_counts)):
            fig, ax = plt.subplots(figsize = figsize)
            sns.heatmap(activepixel_counts[n], cmap='hot',cbar=True,xticklabels=100,yticklabels=100, vmax= vmax, ax=ax, cbar_kws={'label': 'Active pixel counts', 'shrink':cbarsize})
            ax.set_xlabel('Pixel', fontsize=fontsize+10)
            ax.set_ylabel('Pixel', fontsize=fontsize+10)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize,rotation=45)
            ax.figure.axes[-1].yaxis.label.set_size(fontsize+10)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=fontsize)
            ax.axis("equal")
            if save == True:
                plt.savefig(os.path.join(self.save_dir, filename+' '+str(n)+ '.png'), dpi=dpi)

    def DistanceDetect(self, dist_range: int):
        """
        Detect the ratios and the time of the fish entering the reward region.
        """
        pos_count_hour = []
        for n in range(self.time):
            if n == 0:
                df = pd.read_csv(self.actpixel_path, sep='\t', nrows=self.frames1sthr, header=None)
            else:
                df = pd.read_csv(self.actpixel_path, sep='\t', nrows=3600*2*self.fps, skiprows=self.frames1sthr+3600*2*self.fps*(n-1), header=None)
            pos_count = np.zeros((len(df)//200//2,40), np.uint32)
            for m in range(len(df)//200//2): 
                if m == len(df)//200//2 - 1:
                    for j in range(200*m,len(df)//2):
                        empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                        empty[df.iloc[j*2]] += 1
                        erosion = cv2.erode(empty.reshape(self.imgshape[:2]), np.ones((3,3), np.uint8), iterations=2)
                        nb_components, _, stats, centroids = cv2.connectedComponentsWithStats(erosion, connectivity=8)
                        sizes = stats[1:, -1]; centroids = centroids[1:]; nb_components = nb_components - 1
                        com_x = [ int(centroids[i][0]) 
                                                    for i in range(0,nb_components) 
                                                    if sizes[i] >= self.minsize ] 
                        if len(com_x) != 0:
                            for i in range(40):
                                if com_x[0] in dist_range[str(i)]:
                                    pos_count[m][i] += 1
                                try:
                                    if com_x[1] in dist_range[str(i)]:
                                        pos_count[m][i] += 1
                                except: pass
                else:
                    for j in range(200*m,200*(m+1)):
                        empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                        empty[df.iloc[j*2]] += 1
                        erosion = cv2.erode(empty.reshape(self.imgshape[:2]), np.ones((3,3), np.uint8), iterations=2)
                        nb_components, _, stats, centroids = cv2.connectedComponentsWithStats(erosion, connectivity=8)
                        sizes = stats[1:, -1]; centroids = centroids[1:]; nb_components = nb_components - 1
                        com_x = [ int(centroids[i][0]) 
                                                    for i in range(0,nb_components) 
                                                    if sizes[i] >= self.minsize ]
                        if len(com_x) != 0:
                            for i in range(40):
                                if com_x[0] in dist_range[str(i)]:
                                    pos_count[m][i] += 1
                                try:
                                    if com_x[1] in dist_range[str(i)]:
                                        pos_count[m][i] += 1
                                except: pass
            pos_count_hour.append(pos_count)
        return pos_count_hour

    def DistIn1Dim(pos_count, time, fontsize=15, save=False, save_dir=None, name=None): 
        for n in range(time):
            fig, ax = plt.subplots(figsize=(5,10))
            sns.heatmap(data=pos_count[n], vmax=10, cmap='viridis')
            plt.xticks(range(0,41,5),range(0,41,5),fontsize=fontsize, rotation=720)
            plt.yticks(range(0,360,10), range(0,360,10), fontsize=fontsize)
            ax.set_ylabel('Time (10 second)', fontsize=fontsize+5)
            ax.set_xlabel('Distance', fontsize=fontsize+5) 
            if save:
                plt.savefig(os.path.join(save_dir,name+' '+str(n+1)+'.png'), dpi=200)

    def DetectFIRframes(self):
        """
        Detect # of frames of fish in the target area.
        """
        FIRnums = []
        for n in range(self.time):
            if n == 0:
                df = pd.read_csv(self.actpixel_path, sep='\t', nrows=self.frames1sthr, header=None)
            else:
                df = pd.read_csv(self.actpixel_path, sep='\t', nrows=3600*2*self.fps, skiprows=self.frames1sthr+3600*2*self.fps*(n-1), header=None)

            for m in range(len(df)//2//600): 
                if m == len(df)//2//600 - 1:
                    FIRnum = 0
                    for j in range(600*m,len(df)//2): 
                        empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                        empty[df.iloc[j*2]] += 1
                        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(empty.reshape(self.imgshape[0],self.imgshape[1]), connectivity=8)
                        sizes = stats[1:, -1]; nb_components = nb_components - 1
                        img2 = np.zeros((output.shape))
                        img2[output == np.argmax(sizes) + 1] = 1
                        for i in range(len(self.omega_rois)):
                            if img2[ self.omega_rois[i][1]:self.omega_rois[i][1]+self.omega_rois[i][3], self.omega_rois[i][0]:self.omega_rois[i][0]+self.omega_rois[i][2]].sum() > 200:
                                FIRnum += 1
                            else: pass
                    FIRnums.append(FIRnum/(len(df)//2-600*m)*100)
                
                else:
                    FIRnum = 0
                    for j in range(600*m,600*(m+1)):
                        empty = np.zeros(self.imgshape[0]*self.imgshape[1], np.uint8)
                        empty[df.iloc[j*2]] += 1
                        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(empty.reshape(self.imgshape[0],self.imgshape[1]), connectivity=8)
                        sizes = stats[1:, -1]; nb_components = nb_components - 1
                        img2 = np.zeros((output.shape))
                        img2[output == np.argmax(sizes) + 1] = 1
                        for i in range(len(self.omega_rois)):
                            if img2[ self.omega_rois[i][1]:self.omega_rois[i][1]+self.omega_rois[i][3], self.omega_rois[i][0]:self.omega_rois[i][0]+self.omega_rois[i][2]].sum() > 200:
                                FIRnum += 1
                            else: pass 
                    FIRnums.append(FIRnum/600*100)
        return FIRnums

def PlotProbDifFishInRoi(ProbOfFish:list, fontsize=35, ref = np.array([33]*300), colors=[], save=False, save_dir=None, filename=None, dpi=150):
    fig, ax = plt.subplots(len(ProbOfFish),1,figsize=(24,15), sharex=True, sharey=True)
    plt.ylim(0,120)
    plt.xlim(0,300)
    plt.style.use('ggplot')
    plt.yticks(range(0,120,50))
    plt.xlabel('Time(min)', fontsize=fontsize+10)
    plt.rcParams['xtick.labelsize']=fontsize
    plt.rcParams['ytick.labelsize']=fontsize
    for i in range(len(ProbOfFish)):
        ax[i].plot(ProbOfFish[i], label='fish' + str(i+1), color=colors[i])
        ax[i].plot(ref, color='gray',linestyle='dashed')
    ax[2].set_ylabel('Ratio(%)', fontsize=fontsize+10)
    fig.legend(loc = 'upper right', ncol=5, fontsize=fontsize-15, shadow=True, edgecolor='k')
    plt.tight_layout()
    if save:
        plt.savefig(save_dir+filename,dpi=dpi)

def RestructureImg(backgound_img, active_pixel: pd.DataFrame):
    backImg_gray = cv2.cvtColor(backgound_img, cv2.COLOR_RGB2GRAY)
    Img_gray_1D = backImg_gray.reshape(backgound_img.shape[0]*backgound_img.shape[1]).copy()
    for i in range(len(active_pixel)//2):
        Img_gray_1D[active_pixel.iloc[i*2]] = active_pixel.iloc[i*2+1]
    restructuredImg = Img_gray_1D.reshape(backgound_img.shape[:2])
    
    cv2.imshow('Restructured Image', restructuredImg)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
    return restructuredImg