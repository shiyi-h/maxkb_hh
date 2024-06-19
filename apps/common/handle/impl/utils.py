import subprocess
import hashlib
import os

def emf_to_png(input_blob, ext='emf'):
    md5 = hashlib.md5(input_blob).hexdigest()
    temp_dir = os.path.abspath(os.path.dirname(__file__))
    input_name = os.path.join(temp_dir, f"md5.{ext}")
    output_name = os.path.join(temp_dir, md5+'.png')
    with open(input_name, 'wb') as f:
        f.write(input_blob)
    inkscape_command = ['inkscape', '-o', output_name, '--export-png-color-mode=RGB_8', input_name]
    try:
        subprocess.run(inkscape_command, check=True)
        print("convert success")
        with open(output_name, 'rb') as f:
            img = f.read() 
        os.remove(input_name)
        os.remove(output_name)
        return img
    except subprocess.CalledProcessError as e:
        print(f"convert error：{e}")
        return input_blob

# from aspose.imaging import FontSettings, Image as AsImage
# from aspose.imaging.fileformats.png import PngColorType
# from aspose.imaging.imageoptions import EmfRasterizationOptions, WmfRasterizationOptions, PngOptions
# from io import BytesIO
# import os

# BasePath = os.path.abspath('/Users/hh/Documents/')
# FontPath = os.path.join(BasePath,'Fonts')
# FontSettings.set_fonts_folder(FontPath)
# FontSettings.update_fonts()
# ImportOptDict = {"emf":EmfRasterizationOptions(), 'wmf':WmfRasterizationOptions()}
# ExportOpt = PngOptions()
# ExportOpt.color_type = PngColorType.TRUECOLOR_WITH_ALPHA


# def emf_to_png(input_blob, ext='emf'):
#     with AsImage.load(BytesIO(input_blob)) as image:
#         output_blob = BytesIO()
#         export_options = ExportOpt.clone()
#         rasterization_options = ImportOptDict.get(ext)
#         rasterization_options.page_width = float(image.width)
#         rasterization_options.page_height = float(image.height)
#         export_options.vector_rasterization_options = rasterization_options
#         image.save(output_blob, export_options)
#         output_blob.seek(0)
#         return output_blob
    
