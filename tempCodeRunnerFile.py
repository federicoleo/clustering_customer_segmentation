fig, ax = plt.subplots()
  outliers_df.plot(subplots=True, layout=(4,4), kind='box', figsize=(15, 17))
  plt.suptitle('Outliers detection', fontsize=15, y=0.9)
