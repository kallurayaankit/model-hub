def generate_html_report(results, model_name, version):
    html = f"""
    <html>
    <head><title>Evaluation Report: {model_name} v{version}</title></head>
    <body>
        <h1>Evaluation Report</h1>
        <p><b>Model:</b> {model_name} version {version}</p>
        <h2>Metrics</h2>
        <ul>
            <li>Accuracy: {results['accuracy']:.3f}</li>
            <li>ROC AUC: {results['roc_auc']:.3f}</li>
            <li>Latency (ms/sample): {results['latency_ms_per_sample']:.3f}</li>
            <li>Fairness Δ: {results['fairness_delta']:.3f}</li>
            <li>Slice Acc (high): {results['slice_acc_high']:.3f}</li>
            <li>Slice Acc (low): {results['slice_acc_low']:.3f}</li>
        </ul>
        <h2>Decision</h2>
        <p style="font-size:24px; color:{'green' if results['go_no_go'] == 'GO' else 'red'}">{results['go_no_go']}</p>
        <h2>Plots</h2>
        <img src="data:image/png;base64,{results['plot_base64']}" width="600">
    </body>
    </html>
    """
    return html