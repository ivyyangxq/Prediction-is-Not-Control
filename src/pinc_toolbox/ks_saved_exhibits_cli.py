"""Regenerate KS attempt017 exhibits from saved readback CSVs; no simulator imports."""
import argparse
import json
import math
from pathlib import Path
import shutil


def regenerate(input_dir, output_dir):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source, output = Path(input_dir), Path(output_dir)
    # New destination keeps all original result and exhibit files intact.
    output.mkdir(parents=True, exist_ok=False)
    outcomes = pd.read_csv(source / '017_OUTCOMES.csv')
    # A terminal economy failure is one global row with no case/readout.
    # Retain it in every declared case/readout denominator rather than dropping NaNs.
    cells = outcomes[['case','group']].dropna().drop_duplicates()
    failed = outcomes[outcomes.case.isna() | outcomes.group.isna()]
    expanded = []
    for _, failure in failed.iterrows():
        for _, cell in cells.iterrows():
            row = failure.to_dict(); row.update(case=cell.case, group=cell.group,
                decision='technical_unavailable', monitor_alarm=False)
            expanded.append(row)
    outcomes = outcomes[outcomes.case.notna() & outcomes.group.notna()]
    if expanded:
        outcomes = pd.concat([outcomes, pd.DataFrame(expanded)], ignore_index=True)
    categories=['no_alarm_source_withheld','stage2_unresolved','model_involvement','monitor_unavailable','technical_unavailable']
    counts = outcomes.groupby(['case','group','decision'],sort=False).size()
    summary = pd.DataFrame([dict(case=c.case,group=c.group,outcome=category,
        count=int(counts.get((c.case,c.group,category),0)))
        for _,c in cells.iterrows() for category in categories])
    summary.to_csv(output / '017_OUTCOME_SUMMARY.csv', index=False)
    table = []
    for (case, group), rows in outcomes.groupby(['case', 'group'], sort=False):
        counts = rows.decision.value_counts(); n = len(rows)
        k = int(counts.get('model_involvement', 0)); p = k/n; z = 1.95996398454
        den = 1+z*z/n; centre = (p+z*z/(2*n))/den
        half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        table.append(dict(case=case, group=group, planned_n=n,
            completed_n=int((rows.status=='complete').sum()),
            first_alarm_count=int(rows.monitor_alarm.astype(str).str.lower().eq('true').sum()),
            model_report_count=k, unresolved_count=int(counts.get('stage2_unresolved',0)),
            no_alarm_count=int(counts.get('no_alarm_source_withheld',0)),
            monitor_unavailable_count=int(counts.get('monitor_unavailable',0)),
            technical_unavailable_count=int(counts.get('technical_unavailable',0)),
            model_report_pipeline_rate=p,wilson95_lo=max(0,centre-half),wilson95_hi=min(1,centre+half)))
    pd.DataFrame(table).to_csv(output/'017_MAIN_RESULT_TABLE.csv',index=False)
    categories=['no_alarm_source_withheld','stage2_unresolved','model_involvement','monitor_unavailable','technical_unavailable']
    stacked=summary.pivot(index=['case','group'],columns='outcome',values='count').fillna(0).reindex(columns=categories,fill_value=0)
    ax=stacked.plot.barh(stacked=True,figsize=(12,7),color=['#bdbdbd','#f0a202','#d1495b','#555555','#7f7f7f'])
    ax.set_xlabel('Planned macro seeds (128 per case/readout)'); ax.set_title('Saved terminal outcomes: first alarm and final report are distinct')
    ax.figure.tight_layout(); save(ax.figure,output,'017_OUTCOME_STACK',plt)
    economic=pd.read_csv(source/'017_ECONOMIC_RESPONSES.csv')
    means=economic.groupby(['case','group','date'],sort=False)[['unemployment_share_diff','gross_saving_diff','consumption_diff']].mean().reset_index()
    means.to_csv(output/'017_ECONOMIC_RESPONSE_MEANS.csv',index=False)
    fig, axes=plt.subplots(3,2,figsize=(11,8),sharex=True)
    groups=list(economic.group.drop_duplicates());metrics=['unemployment_share_diff','gross_saving_diff','consumption_diff']
    for j,group in enumerate(groups):
        for i,metric in enumerate(metrics):
            ax=axes[i,j]
            for case, rows in means[means.group==group].groupby('case',sort=False):
                ax.plot(rows.date-70,rows[metric],label=case)
            ax.axvline(0,color='.4',ls='--');ax.axhline(0,color='.7',lw=.6);ax.set_ylabel(metric)
            if i==0: ax.set_title(group)
            if i==2: ax.set_xlabel('Periods relative to event')
    axes[0,1].legend(fontsize=6)
    fig.suptitle(f'Saved paired fixed-price PE responses ({economic.seed.nunique()}/128 available macro paths)')
    fig.tight_layout();save(fig,output,'017_ECONOMIC_RESPONSES',plt)
    boundary=pd.read_csv(source/'017_BOUNDARY_READBACK.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,readout in enumerate(['monitor','source']):
        rows=boundary[(boundary.readout==readout)&(boundary.candidate=='N')]
        if readout=='source': rows=rows[rows.source_state!='withheld']
        coverage=rows['observed_'+readout+'_coverage'].dropna()
        axes[0].scatter([j]*len(coverage),coverage,s=6,alpha=.2,label=readout)
        pvalues=rows.p_two_sided.dropna();axes[1].scatter([j]*len(pvalues),pvalues,s=6,alpha=.2)
    for ax in axes: ax.set_xticks([0,1]);ax.set_xticklabels(['monitor','source'])
    axes[0].set_ylabel('Saved whole-window household coverage');axes[1].set_ylabel('Saved 79-slot two-sided p');axes[1].axhline(.025,color='.5',ls='--')
    fig.suptitle('Saved support/bank diagnostics; no new score or threshold');fig.tight_layout();save(fig,output,'017_SUPPORT_BANK_BOUNDARY',plt)
    shutil.copy2(source/'017_READBACK_STATUS.json',output/'017_READBACK_STATUS.json')
    receipt=dict(operation='saved_ks_exhibits',input_dir=str(source.resolve()),output_dir=str(output.resolve()),case_readouts=len(table),macro_seeds=int(outcomes.seed.nunique()),available_economic_paths=int(economic.seed.nunique()),simulation_runs=0,fits=0,calibration_draws=0,scope='Saved corrected attempt017 readback only; no KS scientific adapter is shipped.')
    (output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def save(fig, directory, stem, plt):
    fig.savefig(directory/(stem+'.png'),dpi=180)
    fig.savefig(directory/(stem+'.pdf'))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();print(json.dumps(regenerate(args.input_dir,args.output_dir)))


if __name__=='__main__': main()
