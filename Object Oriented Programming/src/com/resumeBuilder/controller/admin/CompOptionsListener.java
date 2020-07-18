package com.resumeBuilder.controller.admin;

import com.resumeBuilder.view.MainView;
import com.resumeBuilder.view.admin.AdminManagementView;
import com.resumeBuilder.view.admin.mngCompany.*;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class CompOptionsListener implements ActionListener {

    private AdmMngCompanyView mngCompView;

    public CompOptionsListener(AdmMngCompanyView mngCompView) {
        this.mngCompView = mngCompView;
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        try {
            if (e.getActionCommand().equals("<--")) {
                AdminManagementView frame = new AdminManagementView();
                frame.setVisible(true);
                mngCompView.setVisible(false);
            } else if (e.getActionCommand().equals("[->")) {
                MainView frame = new MainView();
                frame.setVisible(true);
                mngCompView.setVisible(false);
            } else if (e.getActionCommand().equals("Search Any Company")) {
                AdmSrchCompByName frame = new AdmSrchCompByName();
                frame.setVisible(true);
                mngCompView.setVisible(false);
            } else if (e.getActionCommand().equals("Display All Companies")) {
                AdmDispAllCompanies frame = new AdmDispAllCompanies();
                frame.setVisible(true);
                mngCompView.setVisible(false);
            } else if (e.getActionCommand().equals("Make Company Account")) {
                AdmComAccCreation ComAccCreationFrame = new AdmComAccCreation();
                ComAccCreationFrame.setVisible(true);
                mngCompView.setVisible(false);
            } else {
                AdmRemoveComByName frame = new AdmRemoveComByName();
                frame.setVisible(true);
                mngCompView.setVisible(false);
            }
        } catch (Exception error) {
            error.printStackTrace();
        }

    }

}
