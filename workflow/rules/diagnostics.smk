"""Diagnostic plots. Not part of the module interface: built only when requested."""


rule plot_heavy_duty_split:
    input:
        levels=rules.fill_heavy_duty_levels.output.levels,
    output:
        figure="<resources>/automatic/diagnostics/heavy_duty_split.png",
    log:
        "<logs>/plot_heavy_duty_split.log",
    conda:
        "../envs/module.yaml"
    params:
        reference_year=config["heavy_duty"]["reference_year"],
    message:
        "Plot heavy-duty vehicle-km per country and year by component, after gap filling."
    script:
        "../scripts/plot_heavy_duty_split.py"
