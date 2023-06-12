#include "./Vulkan/base/VulkanDebug.h"
#include "./RenderDocModules.h"

const save_dir = (__dirname + "./RenderDoc/records/*");

int main()
{
    render_doc.startRecording(save_dir, rener_doc.current_record_filename);
}