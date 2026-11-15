#pragma once

#include <GL/gl.h>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace viewer {
    struct Texture2D {
        GLuint id = 0;
        std::uint32_t width = 0, height = 0;
        std::uint64_t revision = 0;

        void update(const std::vector<std::uint8_t>& rgba, std::uint32_t newWidth, std::uint32_t newHeight,
                    decltype(&glTexImage2D) allocate = &glTexImage2D,
                    decltype(&glTexSubImage2D) upload = &glTexSubImage2D) {
            const std::size_t pixels = static_cast<std::size_t>(newWidth) * newHeight;
            if(!newWidth || !newHeight || newWidth > std::numeric_limits<GLsizei>::max() ||
               newHeight > std::numeric_limits<GLsizei>::max() || pixels > std::numeric_limits<std::size_t>::max()/4 ||
               rgba.size() < pixels*4)
                throw std::runtime_error("Invalid preview texture dimensions or pixel buffer");
            // Errors from earlier UI drawing must not be attributed to this
            // upload. A lost context must not cause an unbounded error loop.
            int cleared = 0;
            while(glGetError() != GL_NO_ERROR && ++cleared < 16) {}
            if(cleared == 16)throw std::runtime_error("OpenGL context is unavailable for preview upload");

            const bool replace = !id || width != newWidth || height != newHeight;
            GLuint target = id;
            if(replace)glGenTextures(1,&target);
            GLenum error = glGetError();
            if(target && error == GL_NO_ERROR) {
                try {
                    glBindTexture(GL_TEXTURE_2D,target);
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
                    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
                    if(replace)
                        allocate(GL_TEXTURE_2D,0,GL_RGBA8,static_cast<GLsizei>(newWidth),static_cast<GLsizei>(newHeight),
                            0,GL_RGBA,GL_UNSIGNED_BYTE,rgba.data());
                    else
                        upload(GL_TEXTURE_2D,0,0,0,static_cast<GLsizei>(newWidth),static_cast<GLsizei>(newHeight),
                            GL_RGBA,GL_UNSIGNED_BYTE,rgba.data());
                    error = glGetError();
                } catch(...) {
                    if(replace)glDeleteTextures(1,&target);
                    glBindTexture(GL_TEXTURE_2D,id);
                    throw;
                }
            }
            if(!target || error != GL_NO_ERROR) {
                if(replace && target)glDeleteTextures(1,&target);
                glBindTexture(GL_TEXTURE_2D,id);
                std::ostringstream text;
                text << "Preview texture " << (replace?"allocation":"upload") << " failed (OpenGL error 0x"
                     << std::hex << error << ')';
                throw std::runtime_error(text.str());
            }
            // Keep the last usable image and its dimensions until the complete
            // new allocation has succeeded. Failed resizes can now be retried.
            if(replace) {
                if(id)glDeleteTextures(1,&id);
                id = target;
                width = newWidth;
                height = newHeight;
            }
            ++revision;
        }

        void destroy() {
            if(id)glDeleteTextures(1,&id);
            id = 0; width = height = 0;
        }
    };
}
