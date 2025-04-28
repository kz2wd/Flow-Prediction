import vtk
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid

import FolderManager
from visualization.saving_file_names import *


def load_grid_with_magnitude(file, magnitude_name):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(file)
    reader.Update()

    array_name = None
    point_data = reader.GetOutput().GetPointData()
    if point_data.HasArray(PREDICTION_ARRAY_NAME):
        array_name = PREDICTION_ARRAY_NAME
    elif point_data.HasArray(TARGET_ARRAY_NAME):
        array_name = TARGET_ARRAY_NAME
    else:
        raise ValueError("No velocity_predict or velocity_target array found.")

    # Compute magnitude
    calculator = vtk.vtkArrayCalculator()
    calculator.SetInputConnection(reader.GetOutputPort())
    calculator.AddVectorArrayName(array_name)
    calculator.SetResultArrayName(magnitude_name)
    calculator.SetFunction(f"mag({array_name})")
    calculator.Update()

    return calculator.GetOutput()


target_model = "A03"
# Step 1: Load both datasets with magnitude
grid_target = load_grid_with_magnitude(FolderManager.FolderManager._generated_data/target_model/TARGET_FILE_NAME, "mag_target")
grid_predict = load_grid_with_magnitude(FolderManager.FolderManager._generated_data/target_model/PREDICTION_FILE_NAME, "mag_predict")

# Step 2: Copy both magnitude arrays into one dataset
# We'll use the geometry and topology of `grid_target`
merged_grid = vtk.vtkStructuredGrid()
merged_grid.DeepCopy(grid_target)

# Add the other magnitude array (from grid_predict) to it
mag_predict_array = grid_predict.GetPointData().GetArray("mag_predict")
merged_grid.GetPointData().AddArray(mag_predict_array)

# Step 3: Compute absolute difference using calculator
diff_calc = vtk.vtkArrayCalculator()
diff_calc.SetInputData(merged_grid)
diff_calc.AddScalarArrayName("mag_target")
diff_calc.AddScalarArrayName("mag_predict")
diff_calc.SetFunction("abs(mag_target - mag_predict)")
diff_calc.SetResultArrayName("mag_diff")
diff_calc.Update()

def create_volume_actor(input_data):
    resampler = vtk.vtkResampleToImage()
    if isinstance(input_data, vtkStructuredGrid):
        producer = vtk.vtkTrivialProducer()
        producer.SetOutput(input_data)
        resampler.SetInputConnection(producer.GetOutputPort())
    else:
        resampler.SetInputConnection(input_data.GetOutputPort())
    resampler.SetSamplingDimensions(100, 100, 100)
    resampler.SetUseInputBounds(True)
    resampler.Update()

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputConnection(resampler.GetOutputPort())

    opacity = vtk.vtkPiecewiseFunction()

    color = vtk.vtkColorTransferFunction()
    color.SetColorSpaceToHSV()
    if isinstance(input_data, vtkStructuredGrid):

        color.AddHSVPoint(0.0, 0.8, 1.0, 1.0)
        color.AddHSVPoint(1.5, 0.667, 1.0, 1.0)  # Blue
        color.AddHSVPoint(3.0, 0.333, 1.0, 1.0)  # Green
        color.AddHSVPoint(5.0, 0.0, 1.0, 1.0)   # Red

        opacity.AddPoint(0.0, 0.4)
        opacity.AddPoint(1.0, 0.6)
        opacity.AddPoint(5.0, 0.95)

    else:
        color.AddHSVPoint(0.0, 0.667, 1.0, 1.0)  # Blue
        color.AddHSVPoint(5.0, 0.333, 1.0, 1.0)  # Green
        color.AddHSVPoint(8.0, 0.0, 1.0, 1.0)  # Red

        opacity.AddPoint(0.0, 0.0)
        opacity.AddPoint(3.0, 0.95)

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    return volume

# Create volume actors
volume_target = create_volume_actor(grid_target)
volume_predict = create_volume_actor(grid_predict)
volume_diff = create_volume_actor(diff_calc)




class VolumeVisu:
    def __init__(self, title, scalar_title, data_source):
        self.title = title
        self.scalar_title = scalar_title
        self.data_source = data_source
        self.volume = create_volume_actor(self.data_source)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.4, 0.65, 0.6)
        self.renderer.AddVolume(self.volume)

    @staticmethod
    def color_tf_to_lookup_table(color_tf, range_min, range_max, num_colors=256):
        lut = vtk.vtkLookupTable()
        lut.SetTableRange(range_min, range_max)
        lut.SetNumberOfTableValues(num_colors)
        lut.Build()

        for i in range(num_colors):
            x = range_min + (range_max - range_min) * i / (num_colors - 1)
            r, g, b = color_tf.GetColor(x)
            lut.SetTableValue(i, r, g, b, 1.0)

        return lut

    def add_scalar_bar(self):
        color_tf = self.volume.GetProperty().GetRGBTransferFunction()

        # You’ll need the scalar range
        if isinstance(self.data_source, vtkStructuredGrid):
            scalar_range = self.data_source.GetPointData().GetScalars().GetRange()
        else:
            scalar_range = self.data_source.GetOutput().GetPointData().GetArray("mag_diff").GetRange()


        # Convert to LUT
        lut = self.color_tf_to_lookup_table(color_tf, *scalar_range)

        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle(self.scalar_title)
        scalar_bar.SetNumberOfLabels(5)
        self.renderer.AddActor2D(scalar_bar)

    def add_title(self):
        # === Add title ===
        title = vtk.vtkTextActor()
        title.SetInput(self.title)
        titleprop = title.GetTextProperty()
        titleprop.SetFontSize(24)
        titleprop.SetBold(1)
        titleprop.SetColor(1, 1, 1)  # white
        self.renderer.AddActor2D(title)

    def add_outline(self):
        # Assuming you have a structured grid (or any vtkDataSet)
        outline_filter = vtk.vtkOutlineFilter()
        if isinstance(self.data_source, vtkStructuredGrid):
            outline_filter.SetInputData(self.data_source)
        else:
            outline_filter.SetInputData(self.data_source.GetOutput())
        outline_filter.Update()

        # Create a mapper and actor for the outline
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline_filter.GetOutputPort())

        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)  # White outline

        # Add to renderer
        self.renderer.AddActor(outline_actor)

# Create renderers
volumes = [VolumeVisu("Target magnitude", "Velocity magnitude", grid_target),
           VolumeVisu("Prediction magnitude", "Velocity magnitude", grid_predict),
           VolumeVisu("Prediction error", "Error", diff_calc),]


shared_camera = vtk.vtkCamera()
render_window = vtk.vtkRenderWindow()
for i, volume in enumerate(volumes):
    volume.add_title()
    volume.add_scalar_bar()
    volume.add_outline()
    volume.renderer.SetViewport(i / 3.0, 0.0, (i + 1) / 3.0, 1.0)
    volume.renderer.SetActiveCamera(shared_camera)
    render_window.AddRenderer(volume.renderer)

render_window.SetSize(1500, 500)

# Setup interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

# Start visualization
render_window.Render()
interactor.Initialize()
interactor.Start()
