"""Instrumented generation metrics; no model or application behavior changes."""
import statistics
import time

class TokenTiming:
    def __init__(self,clock=time.perf_counter):
        self.clock=clock
        self.prompt=True
        self.first=None
        self.last=None
        self.count=0
    def put(self,value):
        # HF generate sends prompt IDs once, then CPU token IDs once per step.
        if self.prompt:
            self.prompt=False
            return
        now=self.clock()
        if self.first is None:self.first=now
        self.last=now
        self.count+=1
    def end(self):
        pass
    def metrics(self,started):
        seconds=self.last-self.first if self.count>1 else None
        tokens=max(0,self.count-1)
        return dict(ttft_ms=(self.first-started)*1000 if self.first is not None else None,
                    decode_seconds=seconds,decode_tokens=tokens,
                    decode_tokens_per_second=tokens/seconds if seconds and seconds>0 else None)

def metric_baseline(report,rows):
    gs=[g for r in rows for g in r.get('generations',[])]
    ds=[g for g in gs if g.get('decode_seconds',0) and g['decode_seconds']>0]
    ttfts=[g['ttft_ms'] for g in gs if g.get('ttft_ms') is not None]
    warm=[r['seconds'] for r in rows[1:]]
    totals=report['totals']
    return dict(model=report['model'],parameters='4B (nominal)',
        precision='NF4 4-bit weights, BF16 compute' if report.get('device')=='cuda' else 'BF16',
        model_footprint_gib=report.get('model_footprint_bytes')/2**30 if report.get('model_footprint_bytes') is not None else None,
        model_cuda_allocated_gib=report.get('model_cuda_allocated_bytes')/2**30 if report.get('model_cuda_allocated_bytes') is not None else None,
        peak_allocated_gib=report.get('peak_allocated_mib')/1024 if report.get('peak_allocated_mib') is not None and report.get('device')=='cuda' else None,
        peak_reserved_gib=report.get('peak_reserved_mib')/1024 if report.get('peak_reserved_mib') is not None and report.get('device')=='cuda' else None,
        ttft_median_ms=statistics.median(ttfts) if ttfts else None,
        decode_tokens_per_second=sum(g['decode_tokens'] for g in ds)/sum(g['decode_seconds'] for g in ds) if ds else None,
        warm_total_latency_median_seconds=statistics.median(warm) if warm else None,
        first_request_seconds=rows[0]['seconds'] if rows else None,
        accuracy=totals['verdict_correct']/totals['claims_total'] if totals.get('claims_total') else None,
        avg_input_tokens=statistics.mean(g['input_tokens'] for g in gs) if gs else None,
        generation_calls=len(gs),ttft_samples=len(ttfts),decode_samples=len(ds),
        measurement='TTFT from model.generate start after tokenization; CPU token streamer synchronizes token delivery. Decode excludes first token; includes EOS. VRAM PyTorch process counters, GiB. Warm latency includes failed requests; accuracy includes missing facts.')

def baseline_markdown(m):
    def number(v,unit,digits=2):return f'{v:.{digits}f} {unit}' if v is not None else 'Not recorded'
    values=[('Model',m['model']),('Parameters',m['parameters']),('Precision',m['precision']),
        ('Model VRAM (allocated after load)',number(m['model_cuda_allocated_gib'],'GiB')),
        ('Model tensor footprint',number(m['model_footprint_gib'],'GiB')),
        ('Peak VRAM (allocated)',number(m['peak_allocated_gib'],'GiB')),
        ('Peak VRAM (reserved)',number(m['peak_reserved_gib'],'GiB')),
        ('TTFT (generation median)',number(m['ttft_median_ms'],'ms')),
        ('Decode speed',number(m['decode_tokens_per_second'],'tok/s')),
        ('Total latency (warm request median)',number(m['warm_total_latency_median_seconds'],'s')),
        ('First request (includes model load)',number(m['first_request_seconds'],'s')),
        ('Accuracy (all gold facts)',number(m['accuracy']*100 if m['accuracy'] is not None else None,'%')),
        ('Avg input tokens (per LLM call)',number(m['avg_input_tokens'],'tokens'))]
    return '| MetricBaseline | Value |\n|---|---|\n'+'\n'.join('| '+k+' | '+v+' |' for k,v in values)+'\n\n'+m['measurement']+'\n'
