from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import wandb.apis.reports as wr
import wandb
assert os.getenv('WANDB_API_KEY')

name_="mustafakeser"
project_="marketing-campaign-wb"
entity_=None
# Get the current date
# current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ","-").replace(":","-")

current_date = os.environ["CUSTOM_DATE"]

# Original string
original_string = "eda-"

# Concatenate the date with the string

run_id = (original_string + current_date)


df = pd.read_csv("train/bank.csv")

run = wandb.init(
    project=project_, 
    name = "1-EDA-report-"+current_date,
    tags = ["EDA"],
    entity=entity_, 
    job_type="exploratory-data-analysis",
    id=run_id)

raw_data_at = wandb.Artifact(
        "marketing-campaign-wb", 
        type="raw_data"
    
    )

stats = df.describe().reset_index()
stats

tbl_train_stats = wandb.Table(data=stats.astype("str"))
wandb.log({"raw_stats_table": tbl_train_stats})

fig, axes = plt.subplots(nrows=9, ncols=2, figsize=(12, 30))
# Flatten the axes for easier indexing
axes = axes.flatten()
# Plot each column's distribution
for i, col in enumerate(df.columns):
    ax = axes[i]
    sns.histplot(df[col],ax=ax,bins=100,kde=True)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(col)
    if i ==1:
        ax.set_xticklabels(df[col].unique(),rotation=30)

# Adjust layout
plt.tight_layout()
fig.suptitle("Distributions",y=1);
plt.savefig("Distributions.png", format="png")
plt.show()

cat_cols = [col for col in df.columns if df[col].dtype=="O"]
num_cols = [col for col in df.columns if df[col].dtype!="O"]
for col in cat_cols:
    plt.figure(figsize=(14,5))
    sns.violinplot(df,x=col,y="balance")
    if i ==0:
        plt.xticks(df[col].unique(),rotation=30)
    plt.title(f"{col} vs balance distributions")
    plt.savefig(f"violinplot_{col}vsbalance_distributions"+".png", format="png")
    plt.show()

plt.figure(figsize=(15,15))
sns.pairplot(df[num_cols+["deposit"]],hue="deposit")
plt.savefig(f"pairplot_num_cols_deposit"+".png", format="png")

fig,ax = plt.subplots(5,2,figsize=(14,24))
axs = ax.flatten()
for i,col in enumerate(cat_cols):
    sns.barplot(y =df[col].value_counts().index ,x =df[col].value_counts().values,ax=axs[i])
    axs[i].set_title(f"{col} counts")
plt.show()
plt.savefig("bar_plots_cat_counts.png", format="png")
fig,ax = plt.subplots(4,2,figsize=(14,14))
axs = ax.flatten()
for i,col in enumerate(num_cols):
#     df[[col,'month']].groupby('month').mean().loc[["jan","feb","mar","may","apr","may","jun","jul","aug","sep","oct","nov","dec"]].plot(ax=axs[i])
    axs[i].plot(df[[col,'month']].groupby('month').mean().loc[["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]].index,df[[col,'month']].groupby('month').mean().loc[["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]].values,label="both")
    axs[i].plot(df[df["deposit"]=="yes"][[col,'month']].groupby('month').mean().loc[["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]].index,df[df["deposit"]=="yes"][[col,'month']].groupby('month').mean().loc[["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]].values,label="yes")
    axs[i].plot(df[df["deposit"]=="no"][[col,'month']].groupby('month').mean().loc[["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]].index,df[df["deposit"]=="no"][[col,'month']].groupby('month').mean().loc[["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]].values,label="no")
    axs[i].set_title(f"{col} monthly chart")
    axs[i].legend()
plt.legend()
fig.suptitle("Montly Deposits",y=1)
plt.tight_layout()
plt.savefig("line_monthly_deposits.png", format="png")
plt.show()
for i,png in enumerate([filename for filename in os.listdir() if filename.endswith('.png')]):
    locals()[f"png_{i}_image"] = wandb.Image(data_or_path=png)
    filename= png.split(".")[0]
    wandb.log({f"png_{filename}": locals()[f"png_{i}_image"]})
    print(f"{filename}")

tbl_df = wandb.Table(data=df.astype(str))
wandb.log({"train_df": tbl_df})
raw_data_at.add(tbl_df, "df")

run.log_artifact(raw_data_at)

run.finish()
run_path = f'mustafakeser/marketing-campaign-wb/{run_id}'  # this is the run id
wapi = wandb.Api()
run = wapi.run(run_path)
run_name = run.name

report = wr.Report(
  entity=entity_,
  project=project_,
  title='EDA',
  description="Automatic EDA Report"
)  
pics = [str(file).split(".png")[0].split("/")[-1]+".png" for file in run.files(names=[]) if (".png" in str(file)) ]
objects = []
for table in ["train_df","raw_stats_table"]:
    header = wr.H1(text=table)
    pg = wr.PanelGrid(
              runsets=[
                  wr.Runset(entity_, project_, "EDA")],
              panels=[ wr.WeavePanelSummaryTable(table)
                  
              ]
            )
    objects.append(header)
    objects.append(pg)
    objects.append(wr.H1(text=" "))

for image_name in pics:
    header = wr.H1(text=image_name.split(".")[0])
    image = wr.Image(f'https://api.wandb.ai/files/{name_}/{project_}/{run_id}/media/images/{image_name}',
                         caption=image_name.split(".")[0]) 
    objects.append(header)
    objects.append(image)
    objects.append(wr.H1(text=" "))
                     

report.blocks = objects
report.save()
print(f"auto-generated report url: {report.url}")
