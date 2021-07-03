from __future__ import annotations
from typing import Callable
import sys
from util import bluePrint, pipePrint, redPrint, stepPrint
from functions import clear_output_folder, create_output_directories
from Dataclasses import PipeConfig
from Dataclasses import PipelineFunction


class Pipeline:
    config: PipeConfig
    steps: list[PipelineFunction]

    def __init__(self, config: PipeConfig):
        self.config = config
        self.steps = []

        self.setup()

    def setup(self):
        """Setup the pipeline"""
        # Create the necessary directories
        self.add(clear_output_folder)
        self.add(create_output_directories)

        # Check if input directory exists
        if not self.config.inputFolder.exists():
            redPrint("The input folder '%s' doesn't exist" %
                     self.config.inputFolder)
            sys.exit()
        if not self.config.classes_txt.exists():
            redPrint("The darknet classes file '%s' doesn't exist" %
                     self.config.classes_txt)
            sys.exit()

    def add(self, function: Callable, *kwargs, **args):
        self.steps.append(
            PipelineFunction(
                pipeline=self,
                function=function,
                kwargs=kwargs,
                args=args
            )
        )

    def execute(self):
        lastRestult = None
        numberOfSteps = len(self.steps)

        for i, step in enumerate(self.steps):
            stepPrint(i+1, numberOfSteps, step.function.__name__)
            # Exectute the steps function
            lastRestult = step.call(input=lastRestult)
            # Done
            pipePrint("Done", style="bold green")
