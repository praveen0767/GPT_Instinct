import traceback
try:
    from utils_inference import run_inference
    res = run_inference('Screenshot 2026-03-14 113016.png')
    print("INFERENCE RESULT:", res)
    
    # Simulate front-end rendering logic
    results = res
    def get_color(conf):
        if conf is None: return "red"
        if conf >= 0.90: return "green"
        elif conf >= 0.70: return "orange"
        else: return "red"

    def metric_card(title, value, conf):
        color = get_color(conf)
        html = f'''
        <div style="border:1px solid #ddd; padding:15px; border-radius:10px; text-align:center; background-color: #f9f9f9;">
            <p style="margin: 0; color: #555; font-size: 14px; font-weight: 600;">{title}</p>
            <h2 style="margin: 5px 0; color: {color};">{value if value is not None else '—'}</h2>
            <div style="background-color: #eee; width: 100%; height: 5px; border-radius: 5px; margin-top: 5px;">
                <div style="background-color: {color}; width: {int((conf if conf is not None else 0.0) * 100)}%; height: 100%; border-radius: 5px;"></div>
            </div>
            <p style="margin-top: 5px; color: gray; font-size: 12px;">Confidence: {conf if conf is not None else 0.0:.2f}</p>
        </div>
        '''
        return html
        
    metric_card("kWh", results.get("kWh"), results.get("kWh_probability", 0.0))
    metric_card("kVAh", results.get("kVAh"), 0.0)
    metric_card("MD kW", results.get("MD_kW"), results.get("decimal_probability", 0.0))
    metric_card("Demand kVA", results.get("Demand_kVA"), 0.0)
    metric_card("Meter Serial", results.get("serial"), results.get("serial_probability", 0.0))
    
    print("UI RENDERING SUCCESSFUL!")
except Exception as e:
    print("CRASHED IN UI SIMULATION!")
    traceback.print_exc()
