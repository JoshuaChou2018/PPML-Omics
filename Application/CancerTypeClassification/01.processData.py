import pandas as pd
import numpy as np
initial_flag=True
#for cancertype in ['BLCA']:
for cancertype in 'ACC BLCA BRCA CESC CHOL COAD COADREAD DLBC ESCA FPPP GBM GBMLGG HNSC KICH KIPAN KIRC KIRP LAML LGG LIHC LUAD LUSC MESO OV PAAD PCPG PRAD READ SARC SKCM STAD STES TGCT THCA THYM UCEC UCS UVM'.split(' '):
    print(cancertype)
    try:
        clin=pd.read_csv('data/gdac.broadinstitute.org_{}.Merge_Clinical.Level_1.2016012800.0.0/{}.clin.merged.txt'.format(cancertype,cancertype),sep='\t')
        clin.index = clin.iloc[:,0]
        clin.columns = clin.iloc[list(clin.index).index('patient.bcr_patient_barcode')]
        rnaseq=pd.read_csv('data/gdac.broadinstitute.org_{}.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0/{}.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt'.format(cancertype,cancertype),sep='\t')
        rnaseq = rnaseq.iloc[1:,:]

        for patient_raw in rnaseq.columns:
            if 'TCGA' in patient_raw:
                print(patient_raw)
                patient = '-'.join(patient_raw.split('-')[0:3]).lower()
                patient_gene_df = rnaseq.loc[:,['Hybridization REF',patient_raw]]
                patient_gene_df.columns = ['features', patient_raw]

                if patient in clin.columns:
                    day2birth = clin.loc['patient.days_to_birth',patient]
                    day2death = clin.loc['patient.days_to_death',patient]
                    print(day2birth,day2death)
                    try:
                        try:
                            dayrecord=np.abs(int(day2birth))+int(day2death)
                        except:
                            dayrecord=np.abs(int(day2birth))
                        new_row = pd.DataFrame({'features':['day2birth','day2death','dayrecord','cancer_type'], patient_raw:[day2birth,day2death,dayrecord, cancertype]})
                        patient_gene_df = pd.concat([new_row, patient_gene_df[:]]).reset_index(drop = True)
                        if initial_flag==True:
                            all_df = patient_gene_df
                            initial_flag=False
                        elif initial_flag==False:
                            all_df = pd.merge(all_df,patient_gene_df, how='outer', on='features')
                    except:
                        print('error')
                else:
                    print('{} with rnaseq but not in clin'.format(patient_raw))
    except:
        print('error')
        
all_df.to_csv('data/data.tsv',sep='\t')
all_df=pd.read_csv('data/data.tsv',sep='\t',index=False)
all_df=all_df.set_index('features')
all_data_dict={'features':list(all_df.index)[4:],'samples':{}}
for sample in all_df.columns:
    if 'TCGA' in sample:
        tmp=list(all_df.loc[:,sample])
        day2birth=tmp[0]
        day2death=tmp[1]
        dayrecord=tmp[2]
        cancertype=tmp[3]
        genes=tmp[4:]
        all_data_dict['samples'][sample]={'day2birth':day2birth,'day2death':day2death,'dayrecord':dayrecord,'cancertype':cancertype,'genes':genes}
np.save('data/data.npy', all_data_dict)

data=np.load('data/data.npy',allow_pickle=True)
data=data.item()
for sample in data['samples']:
    tmp=[float(x) for x in data['samples'][sample]['genes']]
    tmp=np.log10(np.asarray(tmp)+0.0001)
    _min=np.min(tmp)
    _max=np.max(tmp)
    data['samples'][sample]['genes']=(np.asarray(tmp)-_min)/(_max-_min)
np.save('../data/data_norm_log10.npy', data)
