from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import cv2
import os
import time
def euler_to_R(pitch, roll, yaw):
    degrees_to_radians = np.pi / 180.0
    c1 = np.cos(yaw*degrees_to_radians)
    s1 = np.sin(yaw*degrees_to_radians)
    c2 = np.cos(roll*degrees_to_radians)
    s2 = np.sin(roll*degrees_to_radians)
    c3 = np.cos(pitch*degrees_to_radians)
    s3 = np.sin(pitch*degrees_to_radians)
    return np.array([[c1*c2, -s1*c3 + c1*s2*s3, s1*s3 + c1*s2*c3],
                    [s1*c2, c1*c3 + s1*s2*s3, -c1*s3 + s1*s2*c3],
                    [-s2, c2*s3, c2*c3]], np.float32)
      
class VOLDOR_Viewer:

    def __init__(self, slam_instance, screen_size=(1280,960), disp_rel_thresh=0.01, near_plane=0.1, far_plane=1000.0):
        self.slam_instance = slam_instance
        self.fx = slam_instance.fx
        self.fy = slam_instance.fy
        self.cx = slam_instance.cx
        self.cy = slam_instance.cy
        self.w = slam_instance.w
        self.h = slam_instance.h
        self.init_screen_size = screen_size
        self.near_plane = near_plane
        self.far_plane = far_plane
        
        self.K = np.array([[self.fx,0,self.cx], [0,self.fy,self.cy], [0,0,1]], np.float32)
        self.K_inv = np.linalg.inv(self.K)
        
        self.disp_rel_thresh = disp_rel_thresh
        self.depth_thresh = slam_instance.basefocal / (disp_rel_thresh*self.w)
        self.conf_thresh = 0.95

        #self.user_view_t = np.array([0,0,0], np.float32)
        #self.user_view_r = np.array([0,0,0], np.float32)
        self.view_eye_pos = np.array([0,0,10], np.float32)
        self.view_euler_angle = np.array([0,0,0], np.float32)
        self.view_center_pos = np.array([0,0,0], np.float32)
        self.view_box_width = 20

        self.pixel_size = 1
        self.sample_stride = 4
        
        self.hide_cams = 0
        self.follow_cur_cam = False

        self.use_perspective_view = False

        self.mouse_left_down = False
        self.mouse_right_down = False
        self.mouse_perv_x = None
        self.mouse_perv_y = None

        self.cache_points = []
        self.cache_point_colors = []

        self.cache_outdated = True


    def draw_cams(self, alpha=1):
        n_frames = len(self.slam_instance.frames)
        for fid in range(n_frames):
            frame = self.slam_instance.frames[fid]
            cam_center = frame.Tcw[:3,3].reshape(-1)
            if frame.is_keyframe:
                glColor4f(1, 0, 1,alpha)
                glPointSize(5)
            else:
                glColor4f(0, 1, 0,alpha)
                glPointSize(3)
            glBegin(GL_POINTS)
            glVertex3f(-cam_center[0],-cam_center[1],-cam_center[2])
            glEnd()
    
    def draw_edges(self,alpha=1):
        glColor4f(0, 1, 0,alpha)
        glBegin(GL_LINES)
        n_frames = len(self.slam_instance.frames)
        for edges in self.slam_instance.edges:
            if edges.fid1 >= n_frames or edges.fid2 >= n_frames:
                continue
            frame1 = self.slam_instance.frames[edges.fid1]
            frame2 = self.slam_instance.frames[edges.fid2]
            cam_center1 = frame1.Tcw[:3,3].reshape(-1)
            cam_center2 = frame2.Tcw[:3,3].reshape(-1)
            
            #glLineWidth(1)
            
            glVertex3f(-cam_center1[0],-cam_center1[1],-cam_center1[2])
            glVertex3f(-cam_center2[0],-cam_center2[1],-cam_center2[2])
        glEnd()

    def draw_structures(self):
        if not self.cache_outdated and self.cache_points is not None:
            #print(f'using cache {len(self.cache_points)}')
            glColor3f(0.5, 0.5, 0.5)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.cache_points.data)
            if self.cache_point_colors is not None:
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_UNSIGNED_BYTE, 0, self.cache_point_colors.data)
            glPointSize(self.pixel_size)
            glDrawArrays(GL_POINTS, 0, self.cache_points.shape[0])
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        else:
            self.cache_points = []
            self.cache_point_colors = []
            Iy, Ix = np.mgrid[0:self.h:self.sample_stride, 0:self.w:self.sample_stride]
            coords_2d = np.stack([Ix, Iy, np.ones_like(Ix)], axis=2).astype(np.float32)
            coords_2d = coords_2d.reshape(-1, 3)
            coords_3d_shared = np.matmul(self.K_inv, coords_2d.T).T
            
            for fid in self.slam_instance.kf_ids:
                #frame = self.slam_instance.frames[self.slam_instance.fid_cur_spakf]
                frame = self.slam_instance.frames[fid]
            
                coords_3d = coords_3d_shared * frame.get_scaled_depth()[0:self.h:self.sample_stride, 0:self.w:self.sample_stride].reshape(-1,1)
                
                mask = (frame.depth_conf[0:self.h:self.sample_stride, 0:self.w:self.sample_stride] > self.conf_thresh).reshape(-1) & (coords_3d[:,2]<self.depth_thresh)
                coords_3d = coords_3d[mask]
                
                Tcw = np.copy(frame.Tcw)
                coords_3d = np.matmul(Tcw[:3,:3], coords_3d.T).T
                coords_3d = coords_3d + Tcw[:3,3]
                coords_3d *= -1
                coords_3d = np.ascontiguousarray(coords_3d)
                self.cache_points.append(coords_3d)

                #glColor3f(0.5, 0.5, 0.5)
                glEnableClientState( GL_VERTEX_ARRAY )
                glVertexPointer(3, GL_FLOAT, 0, coords_3d.data)
                
                if self.slam_instance.image_loader_sync(fid, no_block=True):
                    points_rgb = self.slam_instance.images_bgri[fid][0:self.h:self.sample_stride, 0:self.w:self.sample_stride].reshape(-1,3)
                    points_rgb = points_rgb[mask]
                    points_rgb[:,[0,1,2]] = points_rgb[:,[2,1,0]] #bgr to rgb
                    points_rgb = np.ascontiguousarray(points_rgb)
                else: #if image loader speed cannot satisfy the slam system
                    points_rgb = np.full_like(coords_3d, 127, dtype=np.uint8)
                self.cache_point_colors.append(points_rgb)
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_UNSIGNED_BYTE, 0, points_rgb.data)
                    
                glPointSize(self.pixel_size)
                glDrawArrays(GL_POINTS, 0, coords_3d.shape[0])
                glDisableClientState(GL_COLOR_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
            if len(self.cache_points) > 0:
                self.cache_points = np.ascontiguousarray(np.concatenate(self.cache_points, axis=0))
                self.cache_point_colors = np.ascontiguousarray(np.concatenate(self.cache_point_colors, axis=0))
                self.cache_outdated = False


    def draw_world(self):
        _,_,self.screen_width,self.screen_height = glGetIntegerv(GL_VIEWPORT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_GREATER)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.9, 0.9, 0.9, 0) # 0.9 can protect your eyes
        glClearDepth(0.0)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self.use_perspective_view:
            #gluPerspective(60, 1, 0,1000) #TODO: add perspective view option. How people select focal length ??
            pass
        else:
            s = self.screen_width/self.screen_height
            glOrtho(-self.view_box_width*s,self.view_box_width*s,-self.view_box_width,self.view_box_width,-1000,1000)
        

        
        view_eye_pos = self.view_eye_pos.copy()
        view_center_pos = self.view_center_pos.copy()
        R = euler_to_R(self.view_euler_angle[0], self.view_euler_angle[1], self.view_euler_angle[2])

        view_eye_pos = np.matmul(R, view_eye_pos) + view_center_pos
        if self.follow_cur_cam:
            #TODO: FIX over-rotate bug
            Tcw_cur = np.linalg.inv(self.slam_instance.Twc_cur)
            view_eye_pos = np.matmul(Tcw_cur[:3,:3], view_eye_pos) + Tcw_cur[:3,3] 
            view_center_pos =  view_center_pos + Tcw_cur[:3,3]
        gluLookAt(-view_eye_pos[0],-view_eye_pos[1],-view_eye_pos[2],
                    -view_center_pos[0],-view_center_pos[1],-view_center_pos[2],
                    0,1,0)
        

        if self.slam_instance._viewer_signal_map_changed:
            self.cache_outdated = True
            self.slam_instance._viewer_signal_map_changed = False

        self.draw_structures()
        if self.hide_cams % 3 == 0:
            self.draw_cams(alpha=1.0)
            self.draw_edges(alpha=1.0)
        elif self.hide_cams % 3 == 1:
            self.draw_edges()
        
        glutSwapBuffers()
        time.sleep(0.01)


    def on_click(self, button, state, x, y):
        # button: left=0, mid=1, right=2, scroll-up=3, scroll-down=4
        # state 0-down 1-up
        self.mouse_perv_x = x
        self.mouse_perv_y = y
        if button==0:
            self.mouse_left_down = True if state==0 else False
        elif button==2:
            self.mouse_right_down = True if state==0 else False
        elif button==3 and state==0:
            if self.use_perspective_view:
                #self.view_eye_pos[2] -= 2.0
                pass
            else:
                #self.view_box_width -= 2.0
                self.view_box_width /= 1.1
        elif button==4 and state==0:
            if self.use_perspective_view:
                #self.view_eye_pos[2] += 2.0
                pass
            else:
                #self.view_box_width += 2.0
                self.view_box_width *= 1.1
        self.view_eye_pos[2] = max(self.view_eye_pos[2], 1)
        self.view_box_width = max(self.view_box_width, 1)
        
            
        
        
    def on_move(self,x,y):
        if self.mouse_perv_x is None or self.mouse_perv_y is None:
            return
        
        if self.mouse_left_down:
            self.view_euler_angle[1] += 0.2*(x-self.mouse_perv_x)
            self.view_euler_angle[0] -= 0.2*(y-self.mouse_perv_y)
            self.view_euler_angle[0] = max(min(self.view_euler_angle[0], 89.999),-89.999)
        if self.mouse_right_down:
            R = euler_to_R(self.view_euler_angle[0], self.view_euler_angle[1], self.view_euler_angle[2])
            if self.follow_cur_cam:
                R = np.matmul(self.slam_instance.Twc_cur[:3,:3].T, R)
            x_offset = (x-self.mouse_perv_x)*self.view_box_width*0.002
            y_offset = (y-self.mouse_perv_y)*self.view_box_width*0.002
            offset = np.matmul(R, np.array([-x_offset, -y_offset,0], np.float32))
            self.view_center_pos += offset
            pass
        self.mouse_perv_x = x
        self.mouse_perv_y = y


    def on_key(self, bkey, x, y):
        key = bkey.decode("utf-8").lower()
        if key == 'q':
            os._exit(1)
        elif key == 'r':
            self.view_center_pos[...] = 0
        elif key == 'w':
            self.pixel_size += 1
        elif key == 's':
            self.pixel_size = max(self.pixel_size - 1, 1)
        elif key == 'a':
            self.sample_stride += 1
            self.cache_outdated = True
            print(f'viewer sample_stride = {self.sample_stride}')
        elif key == 'd':
            self.sample_stride = max(self.sample_stride - 1, 1)
            self.cache_outdated = True
            print(f'viewer sample_stride = {self.sample_stride}')
        elif key == 'h':
            self.hide_cams += 1
        elif key == 'f':
            self.follow_cur_cam = not self.follow_cur_cam
        elif key == 'x':
            self.disp_rel_thresh /= 1.2
            self.depth_thresh = self.slam_instance.basefocal / (self.disp_rel_thresh*self.w)
            self.cache_outdated = True
        elif key == 'z':
            self.disp_rel_thresh *= 1.2
            self.depth_thresh = self.slam_instance.basefocal / (self.disp_rel_thresh*self.w)
            self.cache_outdated = True
        elif key == 'p':
            self.use_perspective_view = not self.use_perspective_view
        elif key == 'm':
            print(self.cache_points.shape)
            print(self.cache_point_colors.shape)
            n_vertices = self.cache_points.shape[0]
            with open('./pc.ply', 'w') as f:
                f.writelines([
                    'ply\n',
                    'format ascii 1.0\n',
                    f'element vertex {n_vertices}\n',
                    'property float x\n',
                    'property float y\n',
                    'property float z\n',
                    'property uchar red\n',
                    'property uchar green\n',
                    'property uchar blue\n'
                    'element face 0\n',
                    'end_header\n'
                ])
                
                for i in range(n_vertices):
                    f.write(f'{self.cache_points[i, 0]} {self.cache_points[i, 1]} {self.cache_points[i, 2]} {self.cache_point_colors[i,0]} {self.cache_point_colors[i,1]} {self.cache_point_colors[i,2]}\n')
        
        

    def start(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
        glutInitWindowSize(self.init_screen_size[0], self.init_screen_size[1])
        
        wind = glutCreateWindow("VOLDOR SLAM VIEWER")
        
        glutDisplayFunc(self.draw_world)
        glutIdleFunc(self.draw_world)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_move)
        glutKeyboardFunc(self.on_key)
        
        
        glutMainLoop()

